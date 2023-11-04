from torch_geometric.datasets import Reddit

dataset = Reddit(root="/home8t/bzx/data/Reddit")
data = dataset[0]
edge_index = data.edge_index
print(edge_index)
print(edge_index.shape)
a = set()
for i in range(edge_index.shape[1]):
    maxx = max(edge_index[1][i], edge_index[0][i])
    minn = min(edge_index[1][i], edge_index[0][i])
    a.add((maxx, minn))
print(len(a))
