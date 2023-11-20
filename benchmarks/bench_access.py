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
    logger.info("bench access hitmap on " + dataset_name)
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
        n_id = minibatch.n_id


# 训练节点的二阶邻居全采样占全图节点的比例
def bench_2hop_ratio(dataset_name: str, num_neighbors: list[int], batch_size: int):
    pass


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
    pass


# 节点访问频率的分布图（对数坐标）
def bench_access_frequency_log(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    pass


# 统计训练节点的邻居的度分布
def bench_train_neighbor_degree_distribution(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    pass


# 训练节点的度分布
def bench_train_node_degree_distribution(
    dataset_name: str, num_neighbors: list[int], batch_size: int
):
    pass


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
    bench_two_batch_overlap(
        dataset_name="livejournal", num_neighbors=[25, 10], batch_size=6000
    )
    logger.info("-" * 20 + "benchmark end" + "-" * 20)
