from collections import deque
import logging
import torch as th
import numpy as np
import torch_geometric as pyg
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.datasets as pyg_dataset

from bgs.graph import CSRGraph

from utils import DatasetCreator

root = "/home8t/bzx/data/"


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


def bench_on_yelp():
    dataset = pyg_dataset.Yelp(root="/home8t/bzx/data/yelp")
    data = dataset[0]
    train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
    print(train_ids.shape)
    print("node nums:", data.num_nodes)
    print("edge nums:", data.num_edges)
    print("is undirected:", data.is_undirected())
    print("train node nums:", train_ids.shape)


def bench_on_pyg_undirected_data():
    edge_index = th.tensor([[0, 1, 1, 2], [1, 0, 2, 3]])
    data = pyg.data.Data(edge_index=edge_index, num_nodes=4)
    print(data.is_undirected())
    print(data.num_edges)


def bench_connected_graph_nums(dataset_name: str):
    logger.info(f"Start benching {dataset_name} dataset")
    data, csr_graph, train_ids = DatasetCreator.pyg_dataset_creator(
        dataset_name, root=root
    )

    connected_graphs = []
    visited = set()

    for node in train_ids:
        if node not in visited:
            connected_graph = set()
            stack = [node]

            while stack:
                current_node = stack.pop()
                if current_node not in visited:
                    visited.add(current_node)
                    connected_graph.add(current_node)

                    neighbors = csr_graph.indices[
                        csr_graph.indptr[current_node] : csr_graph.indptr[
                            current_node + 1
                        ]
                    ]
                    stack.extend(neighbors)

            connected_graphs.append(connected_graph)

    num_connected_graphs = len(connected_graphs)
    logger.info(f"Number of connected graphs: {num_connected_graphs}")
    return num_connected_graphs


if __name__ == "__main__":
    # logging setup
    logger = logging.getLogger("dataset-bench")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log/bench_dataset.log", mode="a")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # bench_on_ogbn_products()
    # bench_on_livejournal()
    # bench_on_pyg_undirected_data()
    # bench_on_reddit()
    bench_on_yelp()
