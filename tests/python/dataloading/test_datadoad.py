import torch as th
from torch.utils.data.dataloader import DataLoader

import torch_geometric as pyg
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Reddit


def test_dataloader_more_epochs():
    t = th.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    loader = DataLoader(t, batch_size=1, shuffle=True)
    for i in range(3):
        print(str(i) + ":epoch")
        for batch in loader:
            print(batch)


def test_neighborloader_type():
    """测试PyG中NeighborLoader迭代返回的minibatch的类型以及其内部的数据结构"""
    dataset = Reddit("/home8t/bzx/data/Reddit")
    data = dataset[0]
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    loader = NeighborLoader(
        data, num_neighbors=[25, 10], batch_size=128, input_nodes=data.train_mask
    )
    for minibatch in loader:
        print(type(minibatch))
        print(type(minibatch.n_id))
        print(minibatch.n_id)


if __name__ == "__main__":
    test_neighborloader_type()
