"""Dataset partitioning algorithm for enhancing the locality of GPU cache"""
import time
import random
import queue
import collections
import os.path as osp

import torch as th
import numpy as np

from ..graph import CSRGraph


def linear_msbfs_train_partition(
    csr_graph: CSRGraph, train_node_ids: th.Tensor, partition_num: int
) -> dict[int, list]:
    """线性多源BFS训练集划分算法，将拓扑上距离更近的训练节点放在一个分区专供特定的GPU训练从而提高数据局部性

    Args:
        csr_graph (CSRGraph): 图的压缩矩阵存储
        train_node_ids (th.Tensor): 训练节点集
        partition_num (int): 分区数量，一般为GPU数量

    Returns:
        dict[int, list]: 训练集分区字典，键为分区号，值为该分区的训练节点列表
    """
    partition_dict = {i: [] for i in range(partition_num)}

    train_node_num = train_node_ids.shape[0]
    train_node_ids = train_node_ids.numpy()
    # 训练节点的mask，用来快速判断节点是否为训练节点
    train_mask: np.ndarray = np.zeros(csr_graph.node_count, dtype=bool)
    train_mask[train_node_ids] = True

    ## TODO 根据训练节点在全图节点列表的id得到训练节点列表中的索引
    train_to_full_dict = {id: index for index, id in enumerate(train_node_ids)}
    pass
    # 生成种子，得到多源BFS的起始节点
    # 从训练节点生成种子
    # ?或许可以从全图生成种子节点，可能会更均匀
    # 是否对结果进行排序
    start_ids = random.sample(range(train_node_num), partition_num)
    start_ids = train_node_ids[start_ids]
    # 根据种子节点得到该种子节点所对应的分区id
    start_partiton_id_dic = {start: index for index, start in enumerate(start_ids)}
    # 层次遍历 BFS遍历
    # 创建分区的当前和下一层队列
    bfs_queues = {
        i: {"now_que": collections.deque(), "next_que": collections.deque()}
        for i in range(partition_num)
    }
    # 创建距离字典{源节点:{目标节点:距离}} 源节点即种子节点
    # partition num个字典，每个字典都储存了种子节点到其它训练节点的距离
    distance_dict = {i: {} for i in start_ids}
    # 初始化BFS队列
    for i, id in enumerate(start_ids):
        bfs_queues[i]["now_que"].append(id)
    # 创建邻居节点set，用来保存每层遍历中节点的所有出邻居并去重（无向图中出邻居就是邻居）
    neighbors_to_next = set()
    # 对于每个分区分别遍历，共遍历partition num次每次遍历都访问所有的训练节点后停止，获取距离字典，种子节点到每个训练节点的距离
    # 分层遍历，获取距离字典
    # 保证加入next que的节点都是未被访问过的
    # 每次从now que弹出节点后就设置visited标记
    # 如果一个节点已经被访问过，那说明其邻居节点已经被加入过next que
    # 访问到一个新节点后，将其没有被访问过的邻居加入next que中
    # 这样可以保证每个新弹出的节点都是未被访问过的，也就不需要再判断并continue
    for i, start_id in enumerate(start_ids):
        layer = 0
        train_visited_count = 0
        # 已访问节点的标记
        train_visited: np.ndarray = np.zeros(train_node_num, dtype=bool)
        full_visited: np.ndarray = np.zeros(csr_graph.node_count, dtype=bool)
        while train_visited_count < train_node_num:
            while bfs_queues[i]["now_que"]:
                # 访问新节点并设置visited标记
                # 只有训练节点才需要加入距离字典和设置visited标记
                nid = bfs_queues[i]["now_que"].popleft()
                full_visited[nid] = True
                if train_mask[nid]:
                    train_visited[train_to_full_dict[nid]] = True
                    train_visited_count += 1
                    distance_dict[start_id][nid] = layer
                # ? 此处应当区分入邻居和出邻居，如果是出邻居，使用csr获取，如果是入邻居，使用csc获取
                out_neighbors = csr_graph.indices[
                    csr_graph.indptr[nid] : csr_graph.indptr[nid + 1]
                ].numpy()
                neighbors_to_next.update(out_neighbors)
            layer += 1
            # print(neighbors_to_next)
            # print(csr_graph.node_count)
            temp_set = {
                neighbor
                for neighbor in neighbors_to_next
                if full_visited[neighbor] == False
            }
            neighbors_to_next.clear()
            neighbors_to_next.update(temp_set)
            # 将未访问的下一层加入next que
            bfs_queues[i]["next_que"].extend(neighbors_to_next)
            # 交换队列
            bfs_queues[i]["now_que"], bfs_queues[i]["next_que"] = (
                bfs_queues[i]["next_que"],
                bfs_queues[i]["now_que"],
            )
            print("一层遍历结束")
        print("一次遍历结束")
        neighbors_to_next.clear()

    # 以上获取到了train node的距离字典
    # 以下开始线性分区
    # 遍历训练集，获取训练集分区
    # TODO 遍历训练集时是否要打乱训练集顺序呢？
    # TODO 将邻居节点的度考虑在内
    # TODO 要考虑每个分区的节点数是否为平均数向下取整或向上取整
    # 向下取整，当填充至最后一个分区时，无需判断就可以将所有的训练节点加入最后一个分区
    ave_part_size = train_node_num // partition_num
    # 统计每个分区中训练节点的数量
    partition_nodes_count = [0] * partition_num
    copy_distance_dict = distance_dict.copy()
    for train_id in train_node_ids:
        # 10000是预先设定的一个很大的距离，是查找失败时返回的默认值
        # 对于每个训练节点，
        start = min(
            copy_distance_dict, key=lambda x: copy_distance_dict[x].get(train_id, 10000)
        )
        # TODO 此处需要精心地设计，如果最小值有多个，min函数只会返回第一个
        # TODO 而事实上，这种情况下需要其他的指标来决定将训练节点划分到哪个分区
        ...  # 这里可能需要添加关于度的指标

        # 获取种子节点对应的分区id
        part_id = start_partiton_id_dic[start]
        partition_dict[part_id].append(train_id)
        partition_nodes_count[part_id] += 1
        # 复制一份distance dic，如果一个分区分配满了，就将其从距离字典中剔除
        # 距离字典中的元素大于1时，删除距离字典的元素
        # 距离字典中保存的每个种子节点到各个训练节点的距离，大于1可以保证最后剩下的训练节点都被分配到最后一个分区
        if partition_nodes_count[part_id] >= ave_part_size:
            if len(copy_distance_dict) > 1:
                del copy_distance_dict[start]

    return partition_dict
