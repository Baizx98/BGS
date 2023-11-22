"""测试节点访问特性"""
import logging
from tqdm import tqdm
import torch as th
import torch_geometric as pyg
import utils
import bgs
import matplotlib.pyplot as plt

root = "/home8t/bzx/data/"


# 节点在一个epoch内的访问次序图
def bench_access_hitmap(dataset_name: str, num_neighbors: list[int], batch_size: int):
    data, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    logger.info("bench access hitmap on " + dataset_name + "_" + str(batch_size))
    # 使用一个GPU进行训练，也就是使用一个NeighborLoader来从完成的训练集中获取每个batch的采样结果
    loader = pyg.loader.NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_ids,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    minibatch_num = train_ids.shape[0] // batch_size
    col = []  # 访问名次
    raw = []  # 节点id
    for i, minibatch in enumerate(loader):
        n_id = minibatch.n_id.tolist()
        for j in range(len(n_id)):
            col.append(i)
            raw.append(n_id[j])
    # 绘制散点图，col为点的列坐标，raw为点的横坐标，col和row是列表
    plt.scatter(col, raw, s=0.1)
    plt.savefig(f"img/access_hitmap_{dataset_name}_{batch_size}.png")


# 训练节点的K阶邻居全采样占全图节点的比例
def bench_khop_ratio(dataset_name: str, num_neighbors: list[int], batch_size: int):
    data, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    loader = pyg.loader.NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_ids,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    k = len(num_neighbors)
    pbar = tqdm(total=train_ids.shape[0])
    node_access = set()
    for minibatch in loader:
        node_access.update(minibatch.n_id.tolist())
        pbar.update(batch_size)
    ratio = len(node_access) / csr_graph.node_count
    logger.info(f"{dataset_name}-{k}-hop ratio: {ratio}")


# 相邻两个batch的采样子图的节点重合率
def bench_two_batch_overlap(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    logger.info("bench two batch overlap on " + dataset_name)
    logger.info("num_neighbors: " + str(num_neighbors))
    logger.info("batch_size: " + str(batch_size))
    data, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    loader = pyg.loader.NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_ids,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )

    pbar = tqdm(total=train_ids.shape[0])

    last_ids = []
    now_ids = []
    overlap_ratio_list = []
    for minibatch in loader:
        last_ids = now_ids
        now_ids = minibatch.n_id.tolist()
        if len(last_ids):
            overlap_ratio = len(set(now_ids) & set(last_ids)) / len(last_ids)
            overlap_ratio_list.append(overlap_ratio)
        pbar.update(batch_size)
    pbar.close()
    # 绘制折线图
    plt.plot(overlap_ratio_list)
    plt.savefig(f"img/two_batch_overlap_{dataset_name}_{batch_size}.png")


# 节点访问频率的分布图
def bench_access_frequency(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    logger.info("bench access frequency on " + dataset_name)
    data, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    loader = pyg.loader.NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_ids,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    freq = th.zeros(csr_graph.node_count)
    pbar = tqdm(total=train_ids.shape[0])
    for minibatch in loader:
        freq[minibatch.n_id] += 1
        pbar.update(batch_size)
    pbar.close()
    # 绘制直方图
    plt.hist(freq.tolist(), bins=100)
    plt.savefig(f"img/access_frequency_{dataset_name}_{batch_size}.png")


# 节点访问频率的分布图（对数坐标）
def bench_access_frequency_log(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    pass


# 统计训练节点的邻居的度分布
def bench_train_neighbor_degree_distribution(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    logger.info("bench train neighbor degree distribution on " + dataset_name)
    data, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    loader = pyg.loader.NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_ids,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    neighbors_set = set()
    for minibatch in loader:
        neighbors_set.update(minibatch.n_id.tolist())
    neighbors_set = neighbors_set - set(train_ids.tolist())
    neighbors_ids = th.tensor(list(neighbors_set))
    neighbors_degrees = csr_graph.out_degrees[neighbors_ids].tolist()
    # 绘制直方图
    plt.hist(neighbors_degrees, bins=100)
    plt.savefig(
        f"img/train_neighbor_degree_distribution_{dataset_name}_{len(num_neighbors)}-hop.png"
    )


# 训练节点的度分布
def bench_train_node_degree_distribution(dataset_name: str):
    logger.info("bench train node degree distribution on " + dataset_name)
    _, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    degrees = csr_graph.out_degrees[train_ids].tolist()
    # 绘制直方图
    plt.hist(degrees, bins=100)
    plt.savefig(f"img/train_node_degree_distribution_{dataset_name}.png")
    plt.clf()  # Clear the current figure

    # 绘制前x%的节点的度之和占全部节点的度之和的比例，横坐标为节点数的百分比，纵坐标为度之和的百分比
    degrees.sort(reverse=True)
    total_degree = sum(degrees)
    x = []
    y = []
    for i in range(1, 101):
        x.append(i)
        y.append(sum(degrees[: int(i * len(degrees) / 100)]) / total_degree)
    plt.plot(x, y)
    plt.savefig(f"img/train_node_degree_distribution_{dataset_name}_ratio.png")


def degree_freq_rank(dataset_name: str, num_neighbors: list[int], batch_size: int):
    logger.info(f"bench degree frequency rank on {dataset_name}")
    data, csr_graph, train_ids = utils.DatasetCreator.pyg_dataset_creator(
        dataset_name, root
    )
    loader = pyg.loader.NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=train_ids,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    freq = th.zeros(csr_graph.node_count)
    degree = csr_graph.out_degrees
    pbar = tqdm(total=train_ids.shape[0])
    for minibatch in loader:
        freq[minibatch.n_id] += 1
        pbar.update(batch_size)
    pbar.close()
    degree_sorted_index = th.argsort(degree, descending=True)
    freq_sorted_index = th.argsort(freq, descending=True)
    degree_rank = th.zeros(csr_graph.node_count, dtype=th.int64)
    freq_rank = th.zeros(csr_graph.node_count, dtype=th.int64)
    degree_rank[degree_sorted_index] = th.arange(csr_graph.node_count)
    degree_rank = degree_rank.tolist()
    freq_rank[freq_sorted_index] = th.arange(csr_graph.node_count)
    freq_rank = freq_rank.tolist()
    sum_loss = 0
    for i in range(csr_graph.node_count):
        sum_loss += abs(degree_rank[i] - freq_rank[i])
    logger.info(f"sum_loss: {sum_loss}")
    ave_loss = sum_loss / csr_graph.node_count
    logger.info(f"ave_loss: {ave_loss}")


if __name__ == "__main__":
    # logging setup
    logger = logging.getLogger("access-bench")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("log/bench_access.log", mode="a")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("-" * 20 + "benchmark setup" + "-" * 20)
    # bench_two_batch_overlap(
    #     dataset_name="livejournal", num_neighbors=[25, 10], batch_size=6000
    # )
    # bench_khop_ratio(
    #     dataset_name="ogbn-products", num_neighbors=[-1, -1], batch_size=1024
    # )
    # bench_access_hitmap("Reddit", [25, 10], 6000)
    # bench_train_node_degree_distribution("ogbn-products")
    # bench_access_frequency("Reddit", [25, 10], 1024)
    # bench_train_neighbor_degree_distribution("livejournal", [-1], 1024)
    degree_freq_rank("livejournal", [25, 10], 1024)
    logger.info("-" * 20 + "benchmark end" + "-" * 20)
