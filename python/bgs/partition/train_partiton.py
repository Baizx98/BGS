"""Dataset partitioning algorithm for enhancing the locality of GPU cache"""
import torch

from ..graph import CSRGraph


def linear_msbfs_train_partition(
    csr_graph: CSRGraph, train_node_ids: torch.Tensor, partition_num: int
) -> dict[int, torch.Tensor]:
    pass
    partition_dict = {0: torch.Tensor([0, 1, 2])}
    return partition_dict
