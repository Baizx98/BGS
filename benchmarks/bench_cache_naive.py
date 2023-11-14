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
from utils import DatasetCreator

dataset_root = "/home8t/bzx/data/"


def bench_naive_on_graph(
    dataset_name: str,
    world_size: int,
    batch_size: int,
    num_neighbors: list[int],
    cache_ratio: float,
    cache_policy: str,
    partition_policy: str,
    gnn_framework: str,
):
    data, csr_graph, train_ids = DatasetCreator.pyg_dataset_creator(
        dataset_name, dataset_root
    )
    # 为数据并行划分训练集
    if partition_policy == "naive":
        logger.info("Naive partition policy")
        train_ids_list: list[th.Tensor] = train_ids.split(
            train_ids.shape[0] // world_size
        )
    else:
        raise NotImplementedError
    # 为每个GPU创建NeighborLoader
    loader_list = [
        NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=train_ids_list[i],
        )
        for i in range(world_size)
    ]

    # 确定缓存策略，将节点缓存至每个GPU
    logger.info(
        cache_policy + "'s cache size: " + str(int(csr_graph.node_count * cache_ratio))
    )

    # 基于度的缓存策略，每个GPU缓存同样的数据
    # PaGraph方案
    if cache_policy == "PaGraph":
        logger.info("PaGraph cache policy")
        cache = CachePagraph(world_size, cache_ratio)
        cache.generate_cache(csr_graph)

    # 基于预采样的缓存策略，每个GPU缓存同样的数据
    # GnnLab方案
    elif cache_policy == "GnnLab":
        logger.info("GnnLab cache policy")
        pre_sampler_epochs = 3
        cache = CacheGnnlab(world_size, cache_ratio)
        cache.generate_cache(csr_graph, loader_list, pre_sampler_epochs)
    # 基于预采样的缓存策略，每个GPU缓存不同的数据
    # GnnLab-partition方案
    elif cache_policy == "GnnLab-partition":
        logger.info("GnnLab-partition cache policy")
        pre_sampler_epochs = 3
        cache = CacheGnnlabPartition(world_size, cache_ratio)
        cache.generate_cache(csr_graph, loader_list, pre_sampler_epochs)
    else:
        raise NotImplementedError

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
    logger.info("Hit ratio list:" + str(hit_ratio_list))


if __name__ == "__main__":
    # logging setup
    logger = logging.getLogger("naive-bench")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log/bench_cache_naive.log", mode="a")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # benchmark setup
    dataset_name = "Reddit"
    world_size = 4
    batch_size = 1024
    num_neighbors = [25, 10]
    cache_ratio = 0.1
    cache_policy = "PaGraph"
    partition_policy = "naive"
    gnn_framework = "pyg"

    logger.info("-" * 20 + "benchmark setup" + "-" * 20)
    logger.info("dataset name: " + dataset_name)
    logger.info("world size: " + str(world_size))
    logger.info("batch size: " + str(batch_size))
    logger.info("num neighbors: " + str(num_neighbors))
    logger.info("cache ratio: " + str(cache_ratio))
    logger.info("cache policy: " + cache_policy)
    logger.info("partition policy: " + partition_policy)
    logger.info("gnn framework: " + gnn_framework)

    bench_naive_on_graph(
        dataset_name=dataset_name,
        world_size=world_size,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        cache_ratio=cache_ratio,
        cache_policy=cache_policy,
        partition_policy=partition_policy,
        gnn_framework=gnn_framework,
    )

    logger.info("-" * 20 + "benchmark end" + "-" * 20)
