"""测试基于训练集划分算法的缓存策略的性能"""
import os
import pickle
import logging

import torch as th
from torch.utils.data import DataLoader
from torch_geometric.datasets.reddit import Reddit
from torch_geometric.loader import NeighborLoader

from bgs.partition import train_partition
from bgs.graph import CSRGraph

from utils import Cache
from utils import CacheGnnlabPartition
from utils import DatasetCreator

dataset_root = "/home8t/bzx/data/"


def bench_linear_mbfs_on_graph(
    dataset_name: str,
    world_size: int,
    batch_size: int,
    num_neighbors: list[int],
    cache_ratio: float,
    cache_policy: str,
    partition_policy: str,
    gnn_framework: str,
    repartition=True,
):
    data, csr_graph, train_ids = DatasetCreator.pyg_dataset_creator(
        dataset_name, dataset_root
    )
    path = (
        "/home8t/bzx/benchmark/train_partition/"
        + dataset_name
        + "_"
        + partition_policy
        + "_"
        + str(world_size)
        + ".pickle"
    )
    os.makedirs(
        os.path.dirname(path), exist_ok=True
    )  # create directory if it doesn't exist
    if os.path.exists(path) and not repartition:
        with open(path, "rb") as file:
            logging.info("load partition from " + path)
            part_dict = pickle.load(file)
            logging.info("load partition successfully")
    else:
        if partition_policy == "linear_msbfs":
            part_dict = train_partition.linear_msbfs_train_partition(
                csr_graph, train_ids, world_size
            )
        elif partition_policy == "linear_msbfs_v2":
            part_dict = train_partition.linear_msbfs_train_partition_v2(
                csr_graph, train_ids, world_size
            )
        elif partition_policy == "combined_msbfs":
            part_dict = train_partition.combined_msbfs_train_partition(
                csr_graph, train_ids, world_size
            )
        else:
            raise NotImplementedError
        with open(path, "wb") as file:
            pickle.dump(part_dict, file)

    # NeighborLoader 为每个GPU生成各自的NeighborLoader，是torch中DataLoader和pyg中NeighborSampler的集成体
    # 每个Neighbor都使用自己分区内的训练节点
    loader_list = [
        NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=part_dict[i],
        )
        for i in range(world_size)
    ]
    # 确定缓存策略，将节点缓存至每个GPU
    # 基于预采样的缓存策略，每个GPU缓存不同的数据
    # GnnLab-partition
    if cache_policy == "GnnLab-partition":
        logger.info("GnnLab-partition cache policy")
        pre_sampler_epochs = 3
        cache = CacheGnnlabPartition(world_size, cache_ratio)
        cache.generate_cache(csr_graph, loader_list, pre_sampler_epochs)
    else:
        raise NotImplementedError
    # 计算缓存冗余度 冗余度=总缓存空间/实际缓存节点数(去重后)
    logger.info("Cache redundancy: " + str(cache.get_reduncy(csr_graph.node_count)))
    # 遍历minibatch，统计命中率
    # hit_count_list = [0 for i in range(world_size)]
    local_hit_count_list = [0 for i in range(world_size)]
    remote_hit_count_list = [0 for i in range(world_size)]
    access_count_list = [0 for i in range(world_size)]
    for gpu_id in range(world_size):
        logging.info(f"Processing GPU {gpu_id}")
        for minibatch in loader_list[gpu_id]:
            # hit_count, access_count = cache.hit_and_access_count(
            #     gpu_id, minibatch.n_id.tolist()
            # )
            (
                local_hit_count,
                remote_hit_count,
                access_count,
            ) = cache.hit_and_access_count_nvlink(gpu_id, minibatch.n_id.tolist())
            # hit_count_list[gpu_id] += hit_count
            local_hit_count_list[gpu_id] += local_hit_count
            remote_hit_count_list[gpu_id] += remote_hit_count
            access_count_list[gpu_id] += access_count
            # logging.info(
            #     f"GPU {gpu_id}: Hit count: {hit_count}, Access count: {access_count}"
            # )
    # 计算命中率
    local_hit_ratio_list = [
        local_hit_count_list[i] / access_count_list[i] for i in range(world_size)
    ]
    remote_hit_count_list = [
        remote_hit_count_list[i] / access_count_list[i] for i in range(world_size)
    ]
    logger.info("Local hit ratio list:" + str(local_hit_ratio_list))
    logger.info("Remote hit ratio list:" + str(remote_hit_count_list))


if __name__ == "__main__":
    # logging setup
    logger = logging.getLogger("msbfs-bench")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log/bench_cache_msbfs.log", mode="a")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # benchmark setup
    dataset_name = "yelp"
    world_size = 4
    batch_size = 1024
    num_neighbors = [25, 10]
    cache_ratio = 0.1
    cache_policy = "GnnLab-partition"
    partition_policy = "combined_msbfs"
    gnn_framework = "pyg"
    repartition = True

    logger.info("-" * 20 + "benchmark setup" + "-" * 20)
    logger.info("dataset name: " + dataset_name)
    logger.info("world size: " + str(world_size))
    logger.info("batch size: " + str(batch_size))
    logger.info("num neighbors: " + str(num_neighbors))
    logger.info("cache ratio: " + str(cache_ratio))
    logger.info("cache policy: " + cache_policy)
    logger.info("partition policy: " + partition_policy)
    logger.info("gnn framework: " + gnn_framework)
    logger.info("repartition: " + str(repartition))

    bench_linear_mbfs_on_graph(
        dataset_name=dataset_name,
        world_size=world_size,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        cache_ratio=cache_ratio,
        cache_policy=cache_policy,
        partition_policy=partition_policy,
        gnn_framework=gnn_framework,
        repartition=True,
    )

    logger.info("-" * 20 + "benchmark end" + "-" * 20)
