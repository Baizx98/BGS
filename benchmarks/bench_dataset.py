from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.datasets as pyg_dataset


def bench_on_ogbn_products():
    """测试ogbn-products数据集的一些特性"""
    dataset = PygNodePropPredDataset("ogbn-products", root="/home8t/bzx/data/")
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    print(type(data))
    print(type(split_idx))
    print(type(train_idx))
    print(data)
    print(split_idx)
    print(train_idx)


def bench_on_livejournal():
    """测试livejournal数据集的加载和使用"""
    edge_index = ...


def bench_on_reddit():
    dataset = pyg_dataset.Reddit(root="/home8t/bzx/data/")
    data = dataset[0]
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    print(train_ids.shape)


if __name__ == "__main__":
    bench_on_reddit()
