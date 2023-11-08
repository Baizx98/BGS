import time

import pytest

import torch as th
from torch import Tensor
import torch_geometric as pyg
from torch_geometric.datasets.reddit import Reddit

from bgs.graph import CSRGraph
from bgs.partition import train_partiton


def test_reddit_train_partiton():
    dataset = Reddit("/home8t/bzx/data/Reddit/")
    data = dataset[0]
    print(type(data.train_mask))
    edge_index = data.edge_index
    print(data.train_mask)
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    print(train_ids)
    csr = CSRGraph(edge_index=edge_index)
    dic = train_partiton.linear_msbfs_train_partition(csr, train_ids, 4)
    print(dic.get(0))


if __name__ == "__main__":
    pass
    # s = time.time()
    # test_reddit_train_partiton()
    # e = time.time()
    # print(s - e)
