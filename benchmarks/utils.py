import torch as th


class Cache:
    def __init__(self, world_size: int) -> None:
        self.cached_ids_list = [set() for i in range(world_size)]

    def cache_nodes_to_gpu(self, gpu_id: int, train_part: list):
        if isinstance(train_part, th.Tensor):
            train_part = train_part.tolist()
        self.cached_ids_list[gpu_id].update(train_part)
        print(train_part[:10])

    def hit_and_access_count(self, gpu_id: int, minibatch: list) -> (int, int):
        if isinstance(minibatch, th.Tensor):
            minibatch = minibatch.tolist()
        access_count = len(minibatch)
        hit_count = len(set(minibatch) & self.cached_ids_list[gpu_id])
        print(minibatch[:10])
        return hit_count, access_count
