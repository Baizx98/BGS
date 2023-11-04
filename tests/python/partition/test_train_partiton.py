import torch
from torch import Tensor

from bgs.graph import CSRGraph
from bgs.partition import train_partiton

edge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
csr = CSRGraph(edge_index)
dic = train_partiton.linear_msbfs_train_partition(csr, torch.tensor([0, 1]), 2)
a = dic.get(0)
a.shape[0]
print(a)
