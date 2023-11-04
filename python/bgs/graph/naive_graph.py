"""Original graphical representation"""
import torch


class NaiveGraph:
    """Graph"""

    def __init__(self) -> None:
        node_ids: torch.Tensor = None
        edge_index: torch.Tensor = None
        node_feat: torch.Tensor = None
        edge_feat: torch.Tensor = None
        edge_ids = None
        lables = None
