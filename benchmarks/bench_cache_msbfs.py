import os
import pickle
import logging

import torch as th
from torch.utils.data import DataLoader
from torch_geometric.datasets.reddit import Reddit
from torch_geometric.loader import NeighborLoader

from bgs.partition import train_partiton
from bgs.graph import CSRGraph

from utils import Cache


def bench_linear_mbfs_on_reddit(new_data=True, world_size=4):
    print("start bench_linear_mbfs_on_reddit")
    dataset = Reddit("/home8t/bzx/data/Reddit")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    node_count = csr_graph.node_count
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    path = "/home8t/bzx/benchmark/train_partition/reddit_" + str(world_size) + ".pickle"
    os.makedirs(
        os.path.dirname(path), exist_ok=True
    )  # create directory if it doesn't exist
    if os.path.exists(path) and not new_data:
        with open(path, "rb") as file:
            logging.info("load partition from " + path)
            part_dict = pickle.load(file)
            logging.info("load partition successfully")
    else:
        part_dict = train_partiton.linear_msbfs_train_partition(
            csr_graph, train_ids, world_size
        )
        with open(path, "wb") as file:
            pickle.dump(part_dict, file)

    # Train_mask_dict
    train_mask_dict = [th.zeros(node_count, dtype=bool) for i in range(world_size)]
    # 将训练集分区的节点tensor还原成mask表示
    for i in range(world_size):
        train_mask_dict[i][part_dict[i]] = True
    # NeighborLoader 为每个GPU生成各自的NeighborLoader，是torch中DataLoader和pyg中NeighborSampler的集成体
    # 每个Neighbor都使用自己分区内的训练节点
    loader_list = [
        NeighborLoader(
            data, num_neighbors=[25, 10], batch_size=128, input_nodes=train_mask_dict[i]
        )
        for i in range(world_size)
    ]
    # 确定缓存策略，将节点缓存至每个GPU
    # 预采样的策略
    freq_list: list[th.Tensor] = [
        th.zeros(node_count, dtype=int) for i in range(world_size)
    ]
    pre_sampler_epochs = 3
    for i in range(world_size):
        for j in range(pre_sampler_epochs):
            for minibatch in loader_list[i]:
                freq_list[i][minibatch.n_id] += 1
    # 对freq_list中的每个tensor进行降序排序，返回排序后的元素的索引
    for i in range(world_size):
        freq_list[i] = th.argsort(freq_list[i], descending=True)
    # 设置每个GPU缓存节点的数量
    cache_ratio = 0.1
    cache_size: int = int(node_count * cache_ratio)
    logging.info("cache_size is " + str(cache_size))
    # 初始化cache
    cache = Cache(world_size)
    for gpu_id in range(world_size):
        # 注意类型转换
        cache.cache_nodes_to_gpu(gpu_id, freq_list[gpu_id][:cache_size].tolist())
        logging.info(str(cache.cached_ids_list[gpu_id]))
    # 遍历minibatch，统计命中率
    hit_count_list = [0 for i in range(world_size)]
    access_count_list = [0 for i in range(world_size)]
    for gpu_id in range(world_size):
        logging.info(f"Processing GPU {gpu_id}")
        for minibatch in loader_list[gpu_id]:
            hit_count, access_count = cache.hit_and_access_count(
                gpu_id, minibatch.n_id.tolist()
            )
            hit_count_list[gpu_id] += hit_count
            access_count_list[gpu_id] += access_count
            logging.info(
                f"GPU {gpu_id}: Hit count: {hit_count}, Access count: {access_count}"
            )
    # 计算命中率
    hit_ratio_list = [
        hit_count_list[i] / access_count_list[i] for i in range(world_size)
    ]
    logging.info(f"Hit ratio list: {hit_ratio_list}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    bench_linear_mbfs_on_reddit(new_data=False)
