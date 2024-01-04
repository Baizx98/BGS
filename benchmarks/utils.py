import os
import logging
from abc import ABC, abstractmethod
from torch._tensor import Tensor
from tqdm import tqdm

import torch as th
import torch.nn.functional as F
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


class DataPlacement(ABC):
    """抽象类，提供数据放置方法的接口"""

    @abstractmethod
    def generate_layout(
        self, world_size: int, cache_size: int, **kwargs
    ) -> list[th.Tensor]:
        """生成每个GPU上要缓存的部分"""
        pass


class NaivePlacement(DataPlacement):
    """Naive数据放置方法，针对训练集不划分的情况，每个GPU中都缓存相同的数据"""

    def generate_layout(self, world_size: int, cache_size: int, **kwargs):
        """
        生成布局列表。

        参数:
            world_size (int): 世界大小。
            cache_size (int): 缓存大小。
            probability (th.Tensor): 概率张量。
            probability_list (list[th.Tensor]): 概率列表。

        返回:
            layout_list (list[list[int]]): 布局列表。
        """
        logger.info("using Naive placement to generate layout")

        probability: th.Tensor = kwargs.get("probability")
        probability_list: list[th.Tensor] = kwargs.get("probability_list")

        # 兼容全局考虑和独自考虑
        # 断言 probability_list 和 probability 不能同时存在，不能同时为None
        assert probability_list is None or probability is None
        assert not (probability_list is None and probability is None)

        if probability is not None:
            sorted_nid = th.argsort(probability, descending=True)
            layout_list = [sorted_nid[:cache_size]] * world_size
        elif probability_list is not None:
            layout_list = []
            for i in range(world_size):
                sorted_nid = th.argsort(probability_list[i], descending=True)
                layout_list.append(sorted_nid[:cache_size])
        else:
            raise NotImplementedError
        return layout_list


class HashPlacement(DataPlacement):
    """哈希数据放置方法"""

    def generate_layout(
        self, world_size: int, cache_size: int, **kwargs
    ) -> list[th.Tensor]:
        """返回每个GPU上要缓存的部分，返回world_size个tensor

        Args:
            access_probability_list (list[th.Tensor]): 不同GPU上的节点访问概率列表

        Returns:
            list[th.Tensor]: 每个GPU上缓存的节点列表
        """
        # TODO 实现
        # 考虑分区和不分区两种实现方式，probability_list和probability不能同时存在
        logger.info("using Hash placement to generate layout")
        probability: th.Tensor = kwargs.get("probability")
        probability_list: list[th.Tensor] = kwargs.get("probability_list")

        assert probability_list is None or probability is None
        assert not (probability_list is None and probability is None)

        if probability is not None:
            sorted_nid = th.argsort(probability, descending=True)
        elif probability_list is not None:
            sum_probability = th.zeros_like(probability_list[0])
            for probability in probability_list:
                sum_probability += probability
            sorted_nid = th.argsort(sum_probability, descending=True)
        layout_list = []
        for gpu_id in range(world_size):
            layout_list.append(
                sorted_nid[gpu_id : cache_size * world_size : world_size]
            )
        return layout_list


class LinearGreedyPlacement(DataPlacement):
    """线性贪心数据放置方法"""

    def generate_layout(
        self, world_size: int, cache_size: int, **kwargs
    ) -> list[th.Tensor]:
        """返回每个GPU上要缓存的部分，返回world_size个tensor

        Args:
            probability_list (list[th.Tensor]): 不同GPU上的节点访问概率列表

        Returns:
            list[th.Tensor]: 每个GPU上缓存的节点列表
        """
        logger.info("using LinearGreedy placement to generate layout")

        probability_list: list[th.Tensor] = kwargs.get("probability_list")

        sum_probability = th.zeros_like(probability_list[0])
        for probability in probability_list:
            sum_probability += probability
        sorted_nid = th.argsort(sum_probability, descending=True)

        cache_set_list = [set() for i in range(world_size)]
        gpu_id_list = [id for id in range(world_size)]
        """ # 多条件比较,如果cache set已满,则返回-1,否则返回概率,如果概率相同,则返回len(cache set)最小的，因为是比较最大值，所以要取反  
            # 总之，max的key函数要返回一个包含两个元素的元祖,第一个元素是概率，第二个元素是len(cache set)的相反数
            # 同时第一个元素要增加一个判断,如果cache set已满,则返回-1,否则返回概率
            # 为了代码简洁,使用了lambda表达式
            # 为了增加key函数的可读性,要添加详细的注释
            # 函数闭包真是妙啊妙啊妙啊
        """

        def custom_compare(nid: int):
            def compare(gpu_id: int):
                cache_set_len = len(cache_set_list[gpu_id])
                probability_value = (
                    probability_list[gpu_id][nid] if cache_set_len < cache_size else -1
                )
                cache_set_value = -len(cache_set_list[gpu_id])
                return probability_value, cache_set_value

            return compare

        for i in range(cache_size * world_size):
            nid = sorted_nid[i]
            selected_gpu_id = max(gpu_id_list, key=custom_compare(nid))
            cache_set_list[selected_gpu_id].add(nid)

        layout_list = []
        for gpu_id in range(world_size):
            layout_list.append(list(cache_set_list[gpu_id]))
        return layout_list


class HeuristicRedundantDataplacement(DataPlacement):
    """基于启发式的冗余数据放置方法
    从零开始构造带有一定冗余的缓存布局"""

    def generate_layout(
        self, world_size: int, cache_size: int, **kwargs
    ) -> list[Tensor]:
        """Generate a layout of nodes to cache on each GPU.
        1. 为每个GPU设置一个分数，分数为当前GPU上节点访问开销的期望
        2. 从空状态开始初始化，每次迭代选择一个分数最大的GPU，将其cpu set中的一个节点移动到local set中
        3. 同时更新其他GPU的nvlink set和sum、score
        4. 从该GPU的cpu set中选择节点后，要判断该节点是否nvlink set中，如果是，要判断是否需要添加这个冗余节点到local set中
        5. 如果都没有添加，就直接将这个节点添加到local set中，同时更新其他GPU的nvlink set和sum、score

        - 整个

        Args:
            world_size (int): The number of GPUs.
            cache_size (int): The number of nodes to cache on each GPU.

        Returns:
            list[Tensor]: A list of tensors containing the nodes to cache on each GPU.
        """
        logger.info("using HeuristicRedundant placement to generate layout")

        probability_list: list[th.Tensor] = kwargs.get("probability_list")
        assert probability_list is not None

        node_count = probability_list[0].shape[0]

        sorted_nid_list = [
            th.argsort(probability, descending=True).tolist()
            for probability in probability_list
        ]
        # 记录每个GPU上节点访问开销的期望
        score_list = [0 for _ in range(world_size)]

        # 对score list排序返回索引，不改变源list
        # 只是暂时把这个语句放在了这里
        sorted_score_index_list = sorted(range(world_size), key=lambda x: score_list[x])
        #
        # 注意操作的元素为int，而不是tensor
        # 集合初始化，初始全部节点均在CPU上
        node_cpu_set_list = [set(range(node_count)) for _ in range(world_size)]
        node_nvlink_set_list = [set() for _ in range(world_size)]
        node_local_set_list = [set() for _ in range(world_size)]
        # cpu & (nvlink | local) =full
        # nvlink & local != empty
        local_p_sum_list = [0 for _ in range(world_size)]
        nvlink_p_sum_list = [0 for _ in range(world_size)]
        cpu_p_sum_list = [0 for _ in range(world_size)]

        left_pointer_list = [0 for _ in range(world_size)]
        right_pointer_list = [0 for _ in range(world_size)]

        # 每个GPU上用于替换冗余节点的候选节点列表，naive方法中的非冗余节点+sorted_nid_list[gpu_id][cache_size:world_size*cache_size]
        # 其中每个元素的长度不同，但都大于(world_size-1)*cache_size,小于world_size*cache_size
        candidate_nodes_list = [[] for _ in range(world_size)]

        return None


class RemoveRedundancyDataplacement(DataPlacement):
    """先使用naive方法初始化cache，再进行迭代来去除部分冗余
    去除冗余时，要从已缓存的节点中概率最小的冗余节点开始去除
    """

    def generate_layout(
        self, world_size: int, cache_size: int, **kwargs
    ) -> list[Tensor]:
        probability_list: list[th.Tensor] = kwargs.get("probability_list")
        assert probability_list is not None
        sorted_nid_list = [
            th.argsort(probability, descending=True) for probability in probability_list
        ]
        pass
        return super().generate_layout(world_size, cache_size, **kwargs)


class AddRedundancyDataplacement(DataPlacement):
    """先使用轮转贪心的方式初始化cache，再从nvlink set中选择概率最大的节点添加到本地冗余，同时弹出local set末尾的节点
    应该关注被弹出的节点是否在其他GPU的nvlink set中，如果在的话，弹出该节点会影响其他GPU的开销分数，如果不在，可以直接弹出
    轮转贪心时，每次都选择开销最大的GPU，将其cpu set中的第一个节点加入local set"""

    def generate_layout(
        self, world_size: int, cache_size: int, **kwargs
    ) -> list[Tensor]:
        """1. 为每个GPU维护一个分数表示访存开销，每轮开始前根据分数对GPU进行升序排序
        2. 每轮选择开销最大的GPU，将其cpu set中的第一个符合条件的节点加入local set，更新分数
        3. 每次选择加入节点时，要判断该节点是否已经被添加到其它GPU的cache中，如果是的话就跳过，继续寻找下一个节点，同时更新nvlink分数
        4. 遍历完一轮后，根据分数重新排序，如果缓存已满，就break，否则继续下一轮
        5. 理论上所有GPU的缓存应该在同一轮中同时变满
        - 分数的设置应该与本GPU的已缓存节点概率和加上nvlink权重因子乘以nvlink set中的节点概率和相关
        - nvlink权重因子的设定要取决于经验了
        - 还要考虑子图大小，子图大小和概率共同决定了数据传输量"""
        return


class Cache(ABC):
    """
    A class for caching nodes to GPU and counting hit and access count.

    Args:
        world_size (int): The number of GPUs.
        cache_ratio (int, optional): The ratio of nodes to cache. Defaults to 0.1.
    """

    def __init__(
        self,
        world_size: int,
        cache_ratio: int,
        layout: DataPlacement,
    ) -> None:
        self.world_size = world_size
        self.cached_ids_list = [set() for i in range(world_size)]
        self.cache_ratio = cache_ratio
        self.layout = layout

    def cache_nodes_to_gpu(self, cache_parts: list[th.Tensor] | list[list]):
        """
        Cache nodes to GPU.

        Args:
            gpu_id (int): The ID of the GPU.
            cache_part (list): The nodes to cache.
        """
        assert self.world_size == len(cache_parts)
        for gpu_id, cache_part in enumerate(cache_parts):
            if isinstance(cache_part, th.Tensor):
                cache_part = cache_part.tolist()
            if isinstance(cache_part[0], th.Tensor):
                cache_part = [x.item() for x in cache_part]
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

    def hit_and_access_count_nvlink(
        self, gpu_id: int, minibatch: list
    ) -> (int, int, int):
        """
        Calculates the local hit count, remote hit count, and access count for a given GPU and minibatch.

        Args:
            gpu_id (int): The ID of the GPU.
            minibatch (list): The minibatch of data.

        Returns:
            tuple: A tuple containing the local hit count, remote hit count, and access count.
        """
        if isinstance(minibatch, th.Tensor):
            minibatch = minibatch.tolist()
        access_count = len(minibatch)
        local_cached_ids = set(self.cached_ids_list[gpu_id])
        local_hit_count = len(set(minibatch) & local_cached_ids)
        all_cached_ids = set()
        for i in range(self.world_size):
            all_cached_ids.update(self.cached_ids_list[i])
        remote_cached_ids = all_cached_ids - local_cached_ids
        remote_hit_count = len(set(minibatch) & remote_cached_ids)
        return local_hit_count, remote_hit_count, access_count

    # 计算GPU上缓存节点的冗余度
    def get_reduncy(self, node_count) -> float:
        """计算冗余度

        Args:
            node_count (int): 总节点数

        Returns:
            float: 冗余度，范围从1到world_size，越大冗余度越高
        """
        all_cached_ids = set()
        for i in range(self.world_size):
            all_cached_ids.update(self.cached_ids_list[i])
        all_capacity = self.cache_ratio * node_count * self.world_size
        return all_capacity / len(all_cached_ids)

    @abstractmethod
    def generate_probability(self, **kwargs):
        """生成节点在GPU或多个GPU上被访问的概率,结果要归一化"""
        raise NotImplementedError

    @abstractmethod
    def generate_cache(self, **kwargs):
        """生成每个GPU上要缓存的部分"""
        raise NotImplementedError


class CachePagraph(Cache):
    def __init__(
        self,
        world_size: int,
        cache_ratio: float,
        layout: DataPlacement = NaivePlacement(),
    ) -> None:
        super().__init__(world_size, cache_ratio, layout)

    def generate_probability(self, **kwargs) -> th.Tensor:
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        out_degrees: th.Tensor = csr_graph.out_degrees
        probability = F.normalize(out_degrees, p=1, dim=0)
        return probability

    def generate_cache(self, **kwargs):
        """
        Generate a cache of nodes with high out degrees for each GPU.

        Args:
            csr_graph (CSRGraph): The graph in CSR format.

        Returns:
            None
        """
        csr_graph: CSRGraph = kwargs.get("csr_graph")

        cache_size = int(csr_graph.node_count * self.cache_ratio)
        probability = self.generate_probability(csr_graph=csr_graph)
        cached_nid_list = self.layout.generate_layout(
            world_size=self.world_size, cache_size=cache_size, probability=probability
        )
        self.cache_nodes_to_gpu(cached_nid_list)


class CacheGnnlab(Cache):
    def __init__(
        self,
        world_size: int,
        cache_ratio: float,
        layout: DataPlacement = NaivePlacement(),
    ) -> None:
        super().__init__(world_size, cache_ratio, layout)

    def generate_probability(self, **kwargs) -> th.Tensor:
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        loader_list: list[NeighborLoader] = kwargs.get("loader_list")
        pre_sampler_epoches: int = kwargs.get("pre_sampler_epoches", 3)

        freq: th.Tensor = th.zeros(csr_graph.node_count, dtype=float)
        for _ in range(pre_sampler_epoches):
            for loader in loader_list:
                ### 解决LOADER传递的问题，直接传递
                for minibatch in loader:
                    freq[minibatch.n_id] += 1
        probability = F.normalize(freq, p=1, dim=0)
        return probability

    def generate_cache(self, **kwargs):
        """
        Generate a cache of nodes for each GPU based on their frequency of access in the given `loader_list`.

        Args:
            csr_graph (CSRGraph): The graph object.
            loader_list (list[NeighborLoader]): A list of NeighborLoader objects for each GPU.
            pre_sampler_epoches (int, optional): The number of epochs to run the pre-sampler for. Defaults to 3.
        """
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        loader_list: list[NeighborLoader] = kwargs.get("loader_list")
        pre_sampler_epoches: int = kwargs.get("pre_sampler_epoches", 3)

        cache_size = int(csr_graph.node_count * self.cache_ratio)
        probability = self.generate_probability(
            csr_graph=csr_graph,
            loader_list=loader_list,
            pre_sampler_epoches=pre_sampler_epoches,
        )
        cached_nid_list = self.layout.generate_layout(
            world_size=self.world_size, cache_size=cache_size, probability=probability
        )
        self.cache_nodes_to_gpu(cached_nid_list)


class CacheGnnlabPartition(CacheGnnlab):
    """
    A class for caching graph nodes on multiple GPUs using GNNLab partitioning strategy.

    Args:
        world_size (int): The number of GPUs to use.

    Attributes:
        cache_ratio (float): The ratio of nodes to cache on each GPU.
    """

    def __init__(
        self,
        world_size: int,
        cache_ratio: float,
        layout: DataPlacement = NaivePlacement(),
    ) -> None:
        super().__init__(world_size, cache_ratio, layout)

    # generate_probability方法使用父类CacheGnnlab的方法

    def generate_cache(self, **kwargs):
        """
        Generate cache for each GPU.

        Args:
            csr_graph (CSRGraph): The graph in CSR format.
            loader_list (list[NeighborLoader]): The list of NeighborLoader objects for each GPU.
            pre_sampler_epoches (int, optional): The number of epochs to run the pre-sampler. Defaults to 3.
        """
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        loader_list: list[NeighborLoader] = kwargs.get("loader_list")
        pre_sampler_epoches: int = kwargs.get("pre_sampler_epoches", 3)

        cache_size = int(csr_graph.node_count * self.cache_ratio)
        probability_list = [
            self.generate_probability(
                csr_graph=csr_graph,
                loader_list=loader_list[i : i + 1],
                pre_sampler_epoches=pre_sampler_epoches,
            )
            for i in range(self.world_size)
        ]
        # 对freq_list中的每个tensor进行降序排序，返回排序后的元素的索引
        # deprecated
        # sorted_nid_list = []
        # for i in range(self.world_size):
        #     sorted_nid_list.append(th.argsort(probability_list[i], descending=True))
        cached_nid_list = self.layout.generate_layout(
            world_size=self.world_size,
            cache_size=cache_size,
            probability_list=probability_list,
        )
        # 设置每个GPU缓存节点的数量
        # deprecated
        # for gpu_id in range(self.world_size):
        #     self.cache_nodes_to_gpu(
        #         gpu_id, sorted_nid_list[gpu_id][:cache_size].tolist()
        #     )
        self.cache_nodes_to_gpu(cached_nid_list)


class CacheMutilMetric(Cache):
    # TODO
    def __init__(
        self,
        world_size: int,
        cache_ratio: float,
        layout: DataPlacement = NaivePlacement(),
    ) -> None:
        super().__init__(world_size, cache_ratio, layout)

    def generate_probability(self, **kwargs) -> th.Tensor:
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        train_ids: th.Tensor = kwargs.get("train_ids")
        device: th.device = kwargs.get("device")

        # TODO 转移到CUDA计算
        # TODO 整数转换为tensor

        csr_graph.to(device)
        train_ids.to(device)

        first_access_probability: th.Tensor = th.zeros(
            csr_graph.node_count,
            dtype=float,
            device=device,
        )
        # TODO 对训练节点初始化权重，这很重要
        pass

        train_mask = th.zeros(csr_graph.node_count, dtype=bool, device=device)
        train_mask[train_ids] = True
        one_hop_neighbor_mask = th.zeros(
            csr_graph.node_count, dtype=bool, device=device
        )

        # 一阶邻居概率计算
        # 从训练节点出发，计算一阶邻居的访问概率
        pbar = tqdm(total=train_ids.shape[0])
        for n_id in train_ids:
            degree = csr_graph.out_degree(n_id)
            if degree == 0:
                continue
            probability = 1 / degree
            neighbors = csr_graph.out_neighbors(n_id)
            one_hop_neighbor_mask[neighbors] = True
            first_access_probability[neighbors] += probability
            pbar.update(1)
        # first_access_probability = th.nn.functional.normalize(
        #     first_access_probability, p=2, dim=0
        # )
        # 二阶邻居概率计算
        # 从一阶邻居出发，计算二阶邻居的访问概率，注意在二阶邻居中可能要排除训练节点
        second_access_probability = th.zeros(
            csr_graph.node_count, dtype=float, device=device
        )
        one_hop_neighbor = one_hop_neighbor_mask.nonzero(as_tuple=False).view(-1)
        pbar = tqdm(total=one_hop_neighbor.shape[0])
        for n_id in one_hop_neighbor:
            degree = csr_graph.out_degree(n_id)
            if degree == 0:
                continue
            probability = 1 / degree
            neighbors = csr_graph.out_neighbors(n_id)
            # if train_mask[neighbor] == False:
            second_access_probability[neighbors] += probability
            pbar.update(1)
        # p为多少呢？范数的设计要考量
        # second_access_probability = th.nn.functional.normalize(
        #     second_access_probability, p=2, dim=0
        # )
        probability = first_access_probability + second_access_probability
        probability = th.nn.functional.normalize(probability, p=2, dim=0)
        return probability

    def generate_cache(self, **kwargs):
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        train_ids = kwargs.get("train_ids")
        device: th.device = kwargs.get("device", th.device("cpu"))

        cache_size = int(self.cache_ratio * csr_graph.node_count)
        probability = self.generate_probability(
            csr_graph=csr_graph, train_ids=train_ids, device=device
        )
        cached_nid_list = self.layout.generate_layout(
            world_size=self.world_size, cache_size=cache_size, probability=probability
        )
        self.cache_nodes_to_gpu(cached_nid_list)


class CacheMutilMetricPartition(CacheMutilMetric):
    # TODO
    pass

    def __init__(
        self,
        world_size: int,
        cache_ratio: float,
        layout: DataPlacement = NaivePlacement(),
    ) -> None:
        super().__init__(world_size, cache_ratio, layout)

    # generate_probability方法使用父类CacheMutilMetric的方法

    def generate_cache(self, **kwargs):
        csr_graph: CSRGraph = kwargs.get("csr_graph")
        train_ids_list: list[th.Tensor] = kwargs.get("train_ids_list")
        device_list: th.device = kwargs.get("device_list", th.device("cpu"))
        # TODO part_dict转为tensor的list
        # TODO gpu list需要处理，最终用多个GPU来完成

        cache_size = int(self.cache_ratio * csr_graph.node_count)
        # TODO 需要DDP处理
        probability_list = [
            self.generate_probability(
                csr_graph=csr_graph,
                train_ids=train_ids_list[i].tolist(),
                device=device_list[i % len(device_list)],
            )
            for i in range(self.world_size)
        ]
        # sorted_nid_list = []
        # for i in range(self.world_size):
        #     sorted_nid_list.append(th.argsort(probability_list[i], descending=True))

        cached_nid_list = self.layout.generate_layout(
            world_size=self.world_size,
            cache_size=cache_size,
            probability_list=probability_list,
        )

        # 设置每个GPU缓存节点的数量
        # for gpu_id in range(self.world_size):
        #     self.cache_nodes_to_gpu(
        #         gpu_id, sorted_nid_list[gpu_id][:cache_size].tolist()
        #     )
        self.cache_nodes_to_gpu(cached_nid_list)


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


def dataplacement_policy_creator(dataplacement_policy: str) -> DataPlacement:
    if dataplacement_policy == "naive":
        return NaivePlacement()
    elif dataplacement_policy == "hash":
        return HashPlacement()
    elif dataplacement_policy == "linear_greedy":
        return LinearGreedyPlacement()
    else:
        raise NotImplementedError
