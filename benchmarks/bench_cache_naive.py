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


def bench_naive_on_reddit(world_size=4):
    dataset = Reddit("/home8t/bzx/data/Reddit")
    data = dataset[0]
    edge_index = data.edge_index
    csr_graph = CSRGraph(edge_index)
    node_count = csr_graph.node_count
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)

    NeighborLoader
