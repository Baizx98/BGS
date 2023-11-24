import time

import pytest

import torch as th
from torch import Tensor
import torch_geometric as pyg
from torch_geometric.datasets.reddit import Reddit
from ogb.nodeproppred import PygNodePropPredDataset

from bgs.graph import CSRGraph
from bgs.partition import train_partition


def test_reddit_train_partiton():
    dataset = Reddit("/home8t/bzx/data/Reddit/")
    data = dataset[0]
    print(type(data.train_mask))
    edge_index = data.edge_index
    print(data.train_mask)
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    print(train_ids)
    csr = CSRGraph(edge_index=edge_index)
    dic = train_partition.linear_msbfs_train_partition(csr, train_ids, 4)
    print(dic.get(0))


def test_sp_ss_by_layer_bfs_on_reddit():
    dataset = Reddit("/home8t/bzx/data/Reddit/")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    start_index = th.randint(0, train_ids.shape[0], (1,))
    print("start index: ", start_index)
    start_id = train_ids[start_index]
    print("start id: ", start_id)
    start = time.time()
    distance = train_partition.sp_of_ss_by_layer_bfs(csr_graph, train_ids, start_id)
    end = time.time()
    print(end - start)
    print(distance[:200])


def test_sp_ss_by_layer_bfs_on_ogbn_products():
    dataset = PygNodePropPredDataset("ogbn-products", root="/home8t/bzx/data/")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    split_idx = dataset.get_idx_split()
    train_ids = split_idx["train"]
    start_index = th.randint(0, train_ids.shape[0], (1,))
    print("start index: ", start_index)
    start_id = train_ids[start_index]
    print("start id: ", start_id)
    start = time.time()
    distance = train_partition.sp_of_ss_by_layer_bfs(csr_graph, train_ids, start_id)
    end = time.time()
    print(end - start)
    print(distance[:200])


def test_sp_ss_by_layer_bfs_on_ogbn_papers100M():
    dataset = PygNodePropPredDataset("ogbn-papers100M", root="/home8t/bzx/data/")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    split_idx = dataset.get_idx_split()
    train_ids = split_idx["train"]
    start_index = th.randint(0, train_ids.shape[0], (1,))
    print("start index: ", start_index)
    start_id = train_ids[start_index]
    print("start id: ", start_id)
    start = time.time()
    distance = train_partition.sp_of_ss_by_layer_bfs(csr_graph, train_ids, start_id)
    end = time.time()
    print(end - start)
    print(distance[:200])


def test_combined_msbfs_train_partition():
    dataset = Reddit("/home8t/bzx/data/Reddit/")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    s = time.time()
    dic = train_partition.combined_msbfs_train_partition(csr_graph, train_ids, 4)
    e = time.time()
    print(e - s, "ah")


if __name__ == "__main__":
    start = time.time()
    # test_sp_ss_by_layer_bfs_on_reddit()
    # test_sp_ss_by_layer_bfs_on_ogbn_products()
    # test_sp_ss_by_layer_bfs_on_ogbn_papers100M()
    test_combined_msbfs_train_partition()
    end = time.time()
    print(end - start)
