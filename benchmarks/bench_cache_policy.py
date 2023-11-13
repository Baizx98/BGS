# 测试不同缓存策略的性能 使用PyG的数据集格式和数据加载器
import torch as th
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset

from bgs.graph import CSRGraph
from utils import Cache
import utils


def bench_cache_on_pagraph_and_gnnlab(dataset_name: str):
    """测试pagraph和gnnlab缓存策略命中率的对比，分析其差异"""
    if dataset_name == "Reddit":
        dataset = Reddit("/home8t/bzx/data/Reddit")
        data = dataset[0]
        edge_index = data.edge_index
        csr_graph = CSRGraph(edge_index)
        node_count = csr_graph.node_count
        train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    elif dataset_name == "ogbn-products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root="/home8t/bzx/data")
        data = dataset[0]
        edge_index = data.edge_index
        csr_graph = CSRGraph(edge_index)
        node_count = csr_graph.node_count
        split_idx = dataset.get_idx_split()
        train_ids = split_idx["train"]

    train_ids_list: list[th.Tensor] = train_ids.split(train_ids.shape[0] // 4)
    loader_list = [
        NeighborLoader(
            data, num_neighbors=[25, 10], batch_size=128, input_nodes=train_ids_list[i]
        )
        for i in range(4)
    ]

    cache_ratio = 0.5
    cache_size: int = int(node_count * cache_ratio)
    cache = Cache(4)
    print("cache size: " + str(cache_size))

    # 基于度的缓存策略，每个GPU缓存同样的数据
    # PaGraph方案
    print("PaGraph cache policy")
    degrees: th.Tensor = csr_graph.out_degrees
    degrees = th.argsort(degrees, descending=True)
    pagraph_cache = degrees[:cache_size].tolist()

    # 基于预采样的缓存策略，每个GPU缓存同样的数据
    # GnnLab方案
    print("GnnLab cache policy")
    freq: th.Tensor = th.zeros(node_count, dtype=int)
    pre_sampler_epochs = 3
    for i in range(4):
        for j in range(pre_sampler_epochs):
            for minibatch in loader_list[i]:
                freq[minibatch.n_id] += 1
    freq = th.argsort(freq, descending=True)
    gnnlab_cache = freq[:cache_size].tolist()

    print(len(set(pagraph_cache) & set(gnnlab_cache)))
    print("缓存重复率：", len(set(pagraph_cache) & set(gnnlab_cache)) / cache_size)

    pagraph_unique = set(pagraph_cache) - set(gnnlab_cache)
    gnnlab_unique = set(gnnlab_cache) - set(pagraph_cache)
    shared = set(pagraph_cache) & set(gnnlab_cache)

    pagraph_hit_count = 0
    gnnlab_hit_count = 0
    shared_hit_count = 0
    access_count = 0
    for i in range(4):
        for minibatch in loader_list[i]:
            n_id = minibatch.n_id.tolist()
            access_count += len(minibatch.n_id)
            pagraph_hit_count += len(set(n_id) & pagraph_unique)
            gnnlab_hit_count += len(set(n_id) & gnnlab_unique)
            shared_hit_count += len(set(n_id) & shared)
    print("pagraph hit rate: ", pagraph_hit_count / access_count)
    print("gnnlab hit rate: ", gnnlab_hit_count / access_count)
    print("shared hit rate: ", shared_hit_count / access_count)


if __name__ == "__main__":
    bench_cache_on_pagraph_and_gnnlab("ogbn-products")
