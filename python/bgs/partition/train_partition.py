"""Dataset partitioning algorithm for enhancing the locality of GPU cache"""
import time
import random
import queue
import collections
import os.path as osp
import logging

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
    # ?如果一个节点已经被访问过，那说明其邻居节点已经被加入过next que
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
            if len(temp_set) == 0:
                print("train id count:", train_visited_count)
                break
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
        if len(temp_set) == 0:
            print("train id count:", train_visited_count)
            break
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


def linear_msbfs_train_partition_v2(
    csr_graph: CSRGraph, train_node_ids: th.Tensor, partition_num: int
) -> dict[int, th.Tensor]:
    """
    Partition the training nodes using linear MSBFS algorithm.

    Args:
        csr_graph: The graph in CSR format.
        train_node_ids: The ids of training nodes.
        partition_num: The number of partitions.

    Returns:
        A dictionary where the key is the partition id and the value is a tensor of node ids in the partition.
    """
    partition_dict: dict[int, list] = {i: [] for i in range(partition_num)}
    distance_dict: dict[int, th.Tensor] = {}

    train_node_num = train_node_ids.shape[0]
    start_ids = random.sample(range(train_node_num), partition_num)
    start_ids = train_node_ids[start_ids]

    for i, start_id in enumerate(start_ids):
        distance = sp_of_ss_by_layer_bfs(csr_graph, train_node_ids, start_id)
        distance_dict[i] = distance

    # linear partition
    ave_part_size = train_node_num // partition_num
    nodes_count_of_partition = [0] * partition_num
    copy_distance_dict = distance_dict.copy()
    for train_id in train_node_ids:
        start_index = min(
            copy_distance_dict, key=lambda x: copy_distance_dict[x][train_id]
        )
        partition_dict[start_index].append(train_id)
        nodes_count_of_partition[start_index] += 1
        if nodes_count_of_partition[start_index] >= ave_part_size:
            if len(copy_distance_dict) > 1:
                del copy_distance_dict[start_index]
    return partition_dict


def sp_of_ss_by_layer_bfs(
    csr_graph: CSRGraph,
    train_node_ids: th.tensor,
    start: th.tensor,
    train_access_ratio: float = 1.0,
    full_access_ratio: float = 1.0,
    layer_limit: int = 100,
    inf=10000000,
) -> th.Tensor:
    """使用层次BFS求单源最短路径"""
    logger = logging.getLogger("partition")
    train_node_num = train_node_ids.shape[0]
    train_mask = th.zeros(csr_graph.node_count, dtype=bool)
    train_mask[train_node_ids] = True
    full_visited = th.zeros(csr_graph.node_count, dtype=bool)
    start_distance = th.zeros(csr_graph.node_count, dtype=int)
    start_distance.fill_(inf)
    layer = 0
    start_distance[start] = 0
    train_visited_count = 1
    full_visited[start.item()] = True
    full_visited_count = 1
    q = collections.deque()
    q.extend([start.item()])
    break_flag = False
    while q:
        layer += 1
        if layer > layer_limit:
            logger.info("到达层数限制")
            break
        layer_node_num = len(q)
        print("layer:", layer)
        neighbors_set = set()
        for _ in range(layer_node_num):
            # nid is a int type
            nid = q.popleft()
            neighbors = csr_graph.out_neighbors(nid)  # pass nid argument
            neighbors_set.update(neighbors.tolist())
        for neighbor in neighbors_set:
            if full_visited[neighbor] == False:
                full_visited[neighbor] = True
                full_visited_count += 1
                if train_mask[neighbor]:
                    start_distance[neighbor] = layer
                    train_visited_count += 1
                    if (
                        train_visited_count >= train_node_num * train_access_ratio
                        or full_visited_count
                        >= csr_graph.node_count * full_access_ratio
                    ):
                        break_flag = True
                        break
                q.append(neighbor)
        if break_flag:
            logger.info("访问节点数达到阈值")
            logger.info(f"遍历完{train_access_ratio*100}%训练节点")
            logger.info("训练节点数：" + str(train_node_num))
            logger.info("访问训练节点数：" + str(train_visited_count))
            logger.info(f"访问全部节点数：{full_visited_count}")
            break
        elif not q:
            logger.info("遍历完连通图")
            logger.info("训练节点数：" + str(train_node_num))
            logger.info("访问训练节点数：" + str(train_visited_count))
            logger.info(f"访问全部节点数：{full_visited_count}")
            break
        else:
            pass

    return start_distance


def combined_msbfs_train_partition(
    csr_graph: CSRGraph, train_node_ids: th.Tensor, partition_num: int
) -> dict[int, list[int]]:
    """结合pagraph的多源BFS训练集划分算法，将拓扑上距离更近的训练节点放在一个分区专供特定的GPU训练从而提高数据局部性"""
    # 首先使用bfs快速得到每个分区的初始初始分区，然后使用pagraph的算法进行后续的划分
    pass
    partition_dict: dict[int, list] = {i: [] for i in range(partition_num)}
    distance_dict: dict[int, th.Tensor] = {}
    train_node_num = train_node_ids.shape[0]
    start_ids_index = random.sample(range(train_node_num), partition_num)
    start_ids = train_node_ids[start_ids_index]
    for i, start_id in enumerate(start_ids):
        print("msbfs start id:", start_id)
        distance = sp_of_ss_by_layer_bfs(
            csr_graph,
            train_node_ids,
            start_id,
            train_access_ratio=1.0 / partition_num,  # 此处的比例需要修改
            full_access_ratio=0.5 / partition_num,
            inf=10000000,
        )
        distance_dict[i] = distance
    # orgin partition linear partition
    ave_part_size = train_node_num // partition_num
    nodes_count_of_partition = [0] * partition_num
    copy_distance_dict = distance_dict.copy()
    print("linear partition")
    for train_id in train_node_ids:
        start_index = min(
            copy_distance_dict, key=lambda x: copy_distance_dict[x][train_id]
        )
        if copy_distance_dict[start_index][train_id] == 10000000:
            continue
        else:
            # 除了距离的第二种指标，还可以考虑度
            # 获取距离该训练节点最近的种子节点的字典索引
            min_distance_index = [
                key
                for key in copy_distance_dict.keys()
                if copy_distance_dict[key][train_id]
                == copy_distance_dict[start_index][train_id]
            ]
            start_index = min(
                min_distance_index, key=lambda x: nodes_count_of_partition[x]
            )
        partition_dict[start_index].append(train_id.item())
        nodes_count_of_partition[start_index] += 1
        if nodes_count_of_partition[start_index] >= ave_part_size:
            if len(copy_distance_dict) > 1:
                del copy_distance_dict[start_index]
    print("nodes_count_of_partition:", nodes_count_of_partition)
    print("ave_part_size:", ave_part_size)
    print("train_node_count:", train_node_num)
    # 维护每个分区的训练节点的全量采样邻居节点，对于未处理的训练节点，对比其全量采样邻居节点和分区内训练节点的全量采样邻居节点的重合度，将其加入重合度高的分区
    # 同时，添加一个负载均衡因子，使得每个分区的训练节点数都尽可能相等
    # TODO 负载均衡因子的计算方式需要修改
    partition_neighbors_set_dict: dict[int, set] = {
        i: set() for i in range(partition_num)
    }
    # 得到初始分区的全量采样邻居节点（训练节点的邻居节点）
    for i in range(partition_num):
        for train_id in partition_dict[i]:
            partition_neighbors_set_dict[i].update(
                csr_graph.out_neighbors(train_id).tolist()
            )
    # 获取未分区的训练节点
    partition_train_ids = set()
    for _, value in partition_dict.items():
        partition_train_ids.update(value)
    print("partition train ids:", len(partition_train_ids))
    unpartition_train_ids = set(train_node_ids.tolist()) - partition_train_ids
    print("unpartition train ids:", len(unpartition_train_ids))
    score_list: list = [0] * partition_num
    for train_id in unpartition_train_ids:
        # 获取该训练节点的全量采样邻居节点
        train_neighbors_set = set(csr_graph.out_neighbors(train_id).tolist())
        for i in range(partition_num):
            # score公式计算，score=交点数量*（分区平均节点数-分区当前节点数）
            score_list[i] = len(
                train_neighbors_set & partition_neighbors_set_dict[i]
            ) * (ave_part_size - nodes_count_of_partition[i])
        max_score_index = score_list.index(max(score_list))  # 分区id
        partition_dict[max_score_index].append(train_id)
        nodes_count_of_partition[max_score_index] += 1
        partition_neighbors_set_dict[max_score_index].update(train_neighbors_set)
    print("nodes_count_of_partition:", nodes_count_of_partition)
    return partition_dict


def combined_msbfs_train_partition_v2():
    # 当最短距离的节点不唯一时，使用最多公共全采样邻居来决定。
    pass
