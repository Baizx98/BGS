import os
import logging
import torch as th
import torch_geometric as pyg
import torch_geometric.datasets as pyg_dataset
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset

from bgs.graph import CSRGraph

# logging setup
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(filename)s %(levelname)s %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class Cache:
    """
    A class for caching nodes to GPU and counting hit and access count.

    Args:
        world_size (int): The number of GPUs.
        cache_ratio (int, optional): The ratio of nodes to cache. Defaults to 0.1.
    """

    def __init__(self, world_size: int, cache_ratio: int = 0.1) -> None:
        self.world_size = world_size
        self.cached_ids_list = [set() for i in range(world_size)]
        self.cache_ratio = cache_ratio

    def cache_nodes_to_gpu(self, gpu_id: int, cache_part: list):
        """
        Cache nodes to GPU.

        Args:
            gpu_id (int): The ID of the GPU.
            cache_part (list): The nodes to cache.
        """
        if isinstance(cache_part, th.Tensor):
            cache_part = cache_part.tolist()
        self.cached_ids_list[gpu_id].update(cache_part)
        logger.info(str(cache_part[:10]))

    def hit_and_access_count(self, gpu_id: int, minibatch: list) -> (int, int):
        """
        Count hit and access count.

        Args:
            gpu_id (int): The ID of the GPU.
            minibatch (list): The nodes to count.

        Returns:
            (int, int): The hit count and access count.
        """
        if isinstance(minibatch, th.Tensor):
            minibatch = minibatch.tolist()
        access_count = len(minibatch)
        hit_count = len(set(minibatch) & self.cached_ids_list[gpu_id])
        return hit_count, access_count

    def hit_and_access_count_nvlink():
        pass


class CachePagraph(Cache):
    def __init__(self, world_size: int, cache_ratio: float) -> None:
        super().__init__(world_size, cache_ratio)

    def generate_cache(self, csr_graph: CSRGraph):
        """
        Generate a cache of nodes with high out degrees for each GPU.

        Args:
            csr_graph (CSRGraph): The graph in CSR format.

        Returns:
            None
        """
        cache_size = int(csr_graph.node_count * self.cache_ratio)
        out_degrees: th.Tensor = csr_graph.out_degrees
        sorted_nid = th.argsort(out_degrees, descending=True)
        for gpu_id in range(self.world_size):
            self.cache_nodes_to_gpu(gpu_id, sorted_nid[:cache_size].tolist())
        return sorted_nid


class CacheGnnlab(Cache):
    def __init__(self, world_size: int, cache_ratio: float) -> None:
        super().__init__(world_size, cache_ratio)

    def generate_cache(
        self,
        csr_graph: CSRGraph,
        loader_list: list[NeighborLoader],
        pre_sampler_epoches: int = 3,
    ):
        """
        Generate a cache of nodes for each GPU based on their frequency of access in the given `loader_list`.

        Args:
            csr_graph (CSRGraph): The graph object.
            loader_list (list[NeighborLoader]): A list of NeighborLoader objects for each GPU.
            pre_sampler_epoches (int, optional): The number of epochs to run the pre-sampler for. Defaults to 3.
        """
        cache_size = int(csr_graph.node_count * self.cache_ratio)
        freq: th.Tensor = th.zeros(csr_graph.node_count, dtype=int)
        for i in range(self.world_size):
            for j in range(pre_sampler_epoches):
                ### 解决LOADER传递的问题，直接传递
                for minibatch in loader_list[i]:
                    freq[minibatch.n_id] += 1
        sorted_nid = th.argsort(freq, descending=True)
        for gpu_id in range(self.world_size):
            self.cache_nodes_to_gpu(gpu_id, sorted_nid[:cache_size].tolist())
        return sorted_nid


class CacheGnnlabPartition(Cache):
    """
    A class for caching graph nodes on multiple GPUs using GNNLab partitioning strategy.

    Args:
        world_size (int): The number of GPUs to use.

    Attributes:
        cache_ratio (float): The ratio of nodes to cache on each GPU.
    """

    def __init__(self, world_size: int, cache_ratio: float) -> None:
        super().__init__(world_size, cache_ratio)

    def generate_cache(
        self,
        csr_graph: CSRGraph,
        loader_list: list[NeighborLoader],
        pre_sampler_epoches: int = 3,
    ):
        """
        Generate cache for each GPU.

        Args:
            csr_graph (CSRGraph): The graph in CSR format.
            loader_list (list[NeighborLoader]): The list of NeighborLoader objects for each GPU.
            pre_sampler_epoches (int, optional): The number of epochs to run the pre-sampler. Defaults to 3.
        """
        cache_size = int(csr_graph.node_count * self.cache_ratio)
        freq_list: list[th.Tensor] = [
            th.zeros(csr_graph.node_count, dtype=int) for i in range(self.world_size)
        ]
        for i in range(self.world_size):
            for j in range(pre_sampler_epoches):
                for minibatch in loader_list[i]:
                    freq_list[i][minibatch.n_id] += 1
        # 对freq_list中的每个tensor进行降序排序，返回排序后的元素的索引
        for i in range(self.world_size):
            freq_list[i] = th.argsort(freq_list[i], descending=True)
        # 设置每个GPU缓存节点的数量
        for gpu_id in range(self.world_size):
            self.cache_nodes_to_gpu(gpu_id, freq_list[gpu_id][:cache_size].tolist())


class CacheMutilMetric(Cache):
    # TODO
    pass


class CacheMutilMetricPartition(Cache):
    # TODO
    pass


class DatasetCreator:
    def __init__(self) -> None:
        pass

    @classmethod
    def pyg_dataset_creator(
        cls, dataset_name: str, dataset_path: str
    ) -> tuple[pyg.data.Data, CSRGraph, th.Tensor]:
        root = os.path.join(dataset_path, dataset_name)
        if dataset_name == "Reddit":
            dataset = pyg_dataset.Reddit(root=root)
            data = dataset[0]
            train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
            return data, CSRGraph(data.edge_index), train_ids
        elif dataset_name == "ogbn-products":
            dataset = PygNodePropPredDataset(dataset_name, root=dataset_path)
            data = dataset[0]
            edge_index = data.edge_index
            split_idx = dataset.get_idx_split()
            train_ids = split_idx["train"]
            return data, CSRGraph(edge_index), train_ids
        elif dataset_name == "livejournal":
            edge_index = th.load(
                os.path.join(dataset_path, dataset_name, "edge_index.pt")
            )
            undirected_edge_index = pyg.utils.to_undirected(edge_index)
            CSRGraph(undirected_edge_index)

            node_num = edge_index.max() + 1
            if (
                os.path.exists(os.path.join(dataset_path, dataset_name, "train.pt"))
                and os.path.exists(os.path.join(dataset_path, dataset_name, "val.pt"))
                and os.path.exists(os.path.join(dataset_path, dataset_name, "test.pt"))
            ):
                train_mask = th.load(
                    os.path.join(dataset_path, dataset_name, "train.pt")
                )
                val_mask = th.load(os.path.join(dataset_path, dataset_name, "val.pt"))
                test_mask = th.load(os.path.join(dataset_path, dataset_name, "test.pt"))
            else:
                train_mask, val_mask, test_mask = split_nodes(
                    node_num, dataset_path, dataset_name
                )
            data = pyg.data.Data(edge_index=edge_index, num_nodes=node_num)
            data.train_mask = train_mask
            train_ids = train_mask.nonzero(as_tuple=False).view(-1)
            return data, CSRGraph(edge_index), train_ids
        elif dataset_name == "ogbn-papers100M":
            dataset = PygNodePropPredDataset(dataset_name, root=dataset_path)
            data = dataset[0]
            edge_index = data.edge_index
            split_idx = dataset.get_idx_split()
            train_ids = split_idx["train"]
            return data, CSRGraph(edge_index), train_ids
        elif dataset_name == "yelp":
            dataset = pyg_dataset.Yelp(root=root)
            data = dataset[0]
            train_ids = data.train_mask.nonzero(as_tuple=False).view(-1)
            return data, CSRGraph(data.edge_index), train_ids
        elif dataset_name == "webgraph":
            # TODO: webgraph dataset
            raise NotImplementedError
        else:
            raise NotImplementedError


def split_nodes(
    node_num: int,
    dataset_path: str,
    dataset_name: str,
    train_ratio: float = 0.65,
    val_ratio: float = 0.1,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    logger.info("splitting nodes for " + dataset_name)
    nids = th.randperm(node_num)
    train_len = int(node_num * train_ratio)
    val_len = int(node_num * val_ratio)
    test_len = node_num - train_len - val_len
    # train mask
    train_mask = th.zeros(node_num, dtype=th.int)
    train_mask[nids[0:train_len]] = 1
    # val mask
    val_mask = th.zeros(node_num, dtype=th.int)
    val_mask[nids[train_len : train_len + val_len]] = 1
    # test mask
    test_mask = th.zeros(node_num, dtype=th.int)
    test_mask[nids[-test_len:]] = 1
    # save
    if dataset_path is not None:
        th.save(train_mask, os.path.join(dataset_path, dataset_name, "train.pt"))
        th.save(val_mask, os.path.join(dataset_path, dataset_name, "val.pt"))
        th.save(test_mask, os.path.join(dataset_path, dataset_name, "test.pt"))
    return train_mask, val_mask, test_mask
