"""测试采样、特征提取、执行阶段的时间开销"""
import time
import torch as th
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

device = th.device("cuda:1")
dataset = pyg.datasets.Reddit("/home8t/bzx/data/Reddit/")
data = dataset[0]
train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=128,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
)


class SAGE(th.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = th.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x


model = SAGE(dataset.num_features, 128, dataset.num_classes, 2).to(device)
total_sampler_time = 0
total_train_time = 0
total_loading_time = 0


def train(epoch):
    global total_sampler_time, total_train_time, total_loading_time
    model.train()
    total_loss = 0
    sampler_start = time.time()
    for batch in train_loader:
        sampler_end = time.time()
        total_sampler_time += sampler_end - sampler_start
        loading_start = time.time()
        batch = batch.to(device)
        loading_end = time.time()
        total_loading_time += loading_end - loading_start
        optimizer.zero_grad()
        train_start = time.time()
        out = model(batch.x, batch.edge_index)
        train_end = time.time()
        total_train_time += train_end - train_start
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
    return total_loss / len(train_loader)


for epoch in range(1, 11):
    optimizer = th.optim.Adam(model.parameters(), lr=0.005)
    start = time.time()
    loss = train(epoch)
    end = time.time()
    print("Epoch: {:02d}, Time: {:.4f}s, Loss: {:.4f}".format(epoch, end - start, loss))

# 绘制采样、特征提取、执行阶段的时间开销的饼图
import matplotlib.pyplot as plt

labels = ["sampler", "train", "loading"]
sizes = [total_sampler_time, total_train_time, total_loading_time]
explode = (0.1, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(
    sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
)
ax1.axis("equal")
plt.savefig("time.png")
