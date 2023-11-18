# 测试不同缓存策略的性能 使用PyG的数据集格式和数据加载器
import torch as th
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
import matplotlib.pyplot as plt

from bgs.graph import CSRGraph
from utils import Cache
from utils import CachePagraph
from utils import CacheGnnlab
from utils import DatasetCreator
import utils

root = "/home8t/bzx/data/"


def bench_cache_on_pagraph_and_gnnlab(dataset_name: str, cache_ratio: float):
    """测试pagraph和gnnlab缓存策略命中率的对比，分析其差异
    测试缓存重复部分和不重复部分的命中率，衡量两种策略的有效性
    """
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
    elif dataset_name == "livejournal":
        data, csr_graph, train_ids = DatasetCreator.pyg_dataset_creator(
            "livejournal", root
        )
        node_count = csr_graph.node_count
    else:
        raise NotImplementedError

    train_ids_list: list[th.Tensor] = train_ids.split(train_ids.shape[0] // 4)
    loader_list = [
        NeighborLoader(
            data, num_neighbors=[25, 10], batch_size=128, input_nodes=train_ids_list[i]
        )
        for i in range(4)
    ]

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
    return (
        cache_ratio,
        pagraph_hit_count / access_count,
        gnnlab_hit_count / access_count,
        shared_hit_count / access_count,
    )


def bench_cahche_hit_ratio_diff(dataset_name: str):
    x = []
    p = []
    g = []
    s = []
    for i in range(1, 10):
        (
            cache_ratio,
            pagraph_hit_rate,
            gnnlab_hit_rate,
            shared_hit_rate,
        ) = bench_cache_on_pagraph_and_gnnlab(dataset_name, i / 10.0)
        x.append(cache_ratio)
        p.append(pagraph_hit_rate)
        g.append(gnnlab_hit_rate)
        s.append(shared_hit_rate)
        print("cache ratio:", cache_ratio)
    plt.plot(x, p, label="pagraph")
    plt.plot(x, g, label="gnnlab")
    plt.plot(x, s, label="shared")
    plt.legend()
    # 保存图片
    plt.savefig(f"cache_hit_ratio_diff.png")


def bench_cache_hit_ratio(dataset_name: str):
    """绘制度缓存策略和预采样缓存策略在缓存不同比例下的缓存内容重复率"""
    if dataset_name == "Reddit":
        data, csr_graph, train_ids = DatasetCreator.pyg_dataset_creator(
            dataset_name, root
        )

    train_ids_list: list[th.Tensor] = train_ids.split(train_ids.shape[0] // 4)
    loader_list = [
        NeighborLoader(
            data,
            num_neighbors=[25, 10],
            batch_size=1024,
            input_nodes=train_ids_list[i],
        )
        for i in range(4)
    ]
    degree_cache = CachePagraph(4, 1)
    d_sorted = degree_cache.generate_cache(csr_graph).tolist()
    presample_cache = CacheGnnlab(4, 1)
    p_sorted = presample_cache.generate_cache(csr_graph, loader_list, 3).tolist()

    for i in range(1):
        print(f"GPU:{i}")
        x_percent = []
        y_repeat = []
        for j in range(1, csr_graph.node_count, 10):
            d_set = set(d_sorted[:j])
            p_set = set(p_sorted[:j])
            x_percent.append(j / csr_graph.node_count)
            y_repeat.append(len(d_set & p_set) / j)
            if j % 10000 == 1:
                print("j:", j)
        plt.plot(x_percent, y_repeat, label="GPU " + str(i))
        # 保存图片
        plt.savefig(f"GPU:cache_hit_ratio.png")


if __name__ == "__main__":
    # bench_cache_on_pagraph_and_gnnlab("ogbn-products")
    # bench_cache_hit_ratio("Reddit")
    bench_cahche_hit_ratio_diff("ogbn-products")
