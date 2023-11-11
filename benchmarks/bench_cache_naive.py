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
from utils import CachePagraph
from utils import CacheGnnlab
from utils import CacheGnnlabPartition


def bench_naive_on_reddit(world_size: int = 4, cache_policy="PaGraph"):
    dataset = Reddit("/home8t/bzx/data/Reddit")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    node_count = csr_graph.node_count
    train_ids: th.Tensor = data.train_mask.nonzero(as_tuple=False).view(-1)

    train_ids_list: list[th.Tensor] = train_ids.split(train_ids.shape[0] // world_size)
    loader_list = [
        NeighborLoader(
            data, num_neighbors=[25, 10], batch_size=6000, input_nodes=train_ids_list[i]
        )
        for i in range(world_size)
    ]

    # 确定缓存策略，将节点缓存至每个GPU
    cache_ratio = 0.1
    cache_size: int = int(node_count * cache_ratio)
    cache = Cache(world_size)
    logging.info("cache size: " + str(cache_size))

    # 基于度的缓存策略，每个GPU缓存同样的数据
    # PaGraph方案
    if cache_policy == "PaGraph":
        logging.info("PaGraph cache policy")
        cache_pagraph = CachePagraph(world_size, cache_ratio)
        cache_pagraph.generate_cache(csr_graph)

    # 基于预采样的缓存策略，每个GPU缓存同样的数据
    # GnnLab方案
    if cache_policy == "GnnLab":
        logging.info("GnnLab cache policy")
        pre_sampler_epochs = 3
        cache_gnnlab = CacheGnnlab(world_size, cache_ratio)
        cache_gnnlab.generate_cache(csr_graph, loader_list, pre_sampler_epochs)
    # 基于预采样的缓存策略，每个GPU缓存不同的数据
    # GnnLab-partition方案
    if cache_policy == "GnnLab-partition":
        logging.info("GnnLab-partition cache policy")
        pre_sampler_epochs = 3
        cache_gnnlab_partition = CacheGnnlabPartition(world_size, cache_ratio)
        cache_gnnlab_partition.generate_cache(
            csr_graph, loader_list, pre_sampler_epochs
        )

    # 计算命中率
    hit_count_list = [0 for i in range(world_size)]
    access_count_list = [0 for i in range(world_size)]
    for gpu_id in range(world_size):
        for minibatch in loader_list[gpu_id]:
            hit_count, access_count = cache.hit_and_access_count(gpu_id, minibatch.n_id)
            hit_count_list[gpu_id] += hit_count
            access_count_list[gpu_id] += access_count
            # logging.info(
            #     f"GPU {gpu_id}:Hit count:{hit_count}, Access count:{access_count}"
            # )
    hit_ratio_list = [
        hit_count_list[i] / access_count_list[i] for i in range(world_size)
    ]
    logging.info(f"Hit ratio list:{hit_ratio_list}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    bench_naive_on_reddit(cache_policy="PaGraph")
