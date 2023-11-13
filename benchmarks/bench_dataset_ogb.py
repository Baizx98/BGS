from ogb.nodeproppred import PygNodePropPredDataset

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
