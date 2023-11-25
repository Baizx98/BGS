from collections import deque
import torch as th
import numpy as np
import torch_geometric as pyg
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.datasets as pyg_dataset

from bgs.graph import CSRGraph


def bench_on_ogbn_products():
    """测试ogbn-products数据集的一些特性"""
    dataset = PygNodePropPredDataset("ogbn-products", root="/home8t/bzx/data/")
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]
    csr_graph = CSRGraph(data.edge_index)
    num_nodes = csr_graph.node_count
    start_node = 0
    distances = np.full(num_nodes, -1)  # 初始化距离数组，-1 表示不可达
    distances[start_node] = 0  # 起始节点距离设为 0

    queue = deque([start_node])  # 使用双端队列作为 BFS 遍历的数据结构
    count = 1
    while queue:
        node = queue.popleft()

        # 获取当前节点的邻居节点
        neighbors = csr_graph.indices[
            csr_graph.indptr[node] : csr_graph.indptr[node + 1]
        ]
        for neighbor in neighbors:
            if distances[neighbor] == -1:  # 如果邻居节点尚未访问过
                distances[neighbor] = distances[node] + 1  # 更新邻居节点的距离
                queue.append(neighbor)  # 将邻居节点加入队列
                count += 1
        if count % 1000 == 0:
            print(count)
    print(count)
    return distances


def bench_on_livejournal():
    """测试livejournal数据集的加载和使用"""
    edge_index = th.load("/home8t/bzx/data/livejournal/edge_index.pt")
    print(edge_index.shape)
    num_nodes = edge_index.max() + 1
    data = pyg.data.Data(edge_index=edge_index, num_nodes=num_nodes)
    print("livejournal is undirected?", data.is_undirected())


def bench_on_reddit():
    dataset = pyg_dataset.Reddit(root="/home8t/bzx/data/Reddit")
    data = dataset[0]
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    print(train_ids.shape)
    print("node nums:", data.num_nodes)
    print("edge nums:", data.num_edges)
    print("is undirected:", data.is_undirected())


def bench_on_pyg_undirected_data():
    edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 3]])
    data = pyg.data.Data(edge_index=edge_index, num_nodes=4)
    print(data.is_undirected())
    print(data.num_edges)


def bench_connected_graph_nums(dataset_name: str):
    pass


if __name__ == "__main__":
    # bench_on_ogbn_products()
    # bench_on_livejournal()
    # bench_on_pyg_undirected_data()
    bench_on_reddit()
