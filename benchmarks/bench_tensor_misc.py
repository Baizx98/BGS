import random
import time
import torch as th


import matplotlib.pyplot as plt


def bench_intersection(n: int):
    ten_a = th.randint(n, (n,))
    set_a = set(random.sample(range(n), n))
    ten_b = th.randint(n, (n,))
    set_b = set(random.sample(range(n), n))
    start_ten = time.time()
    ten_mask = th.isin(ten_a, ten_b)
    # 获取tenmask的为True的元素个数
    ten_count = th.sum(ten_mask)
    end_ten = time.time()
    ten_time = end_ten - start_ten
    start_set = time.time()
    set_count = len(set_a.intersection(set_b))
    end_set = time.time()
    set_time = end_set - start_set
    return ten_time, set_time


if __name__ == "__main__":
    print("start")
    n_values = [1000, 10000, 100000, 1000000]
    ten_times = []
    set_times = []
    for n in n_values:
        ten_time_sum = 0
        set_time_sum = 0
        print("start")
        for i in range(3):
            ten_time, set_time = bench_intersection(n)
            ten_time_sum += ten_time
            set_time_sum += set_time
        ten_times.append(ten_time_sum / 3)
        set_times.append(set_time_sum / 3)
    print("start")
    plt.plot(n_values, ten_times, label="ten count time")
    plt.plot(n_values, set_times, label="set count time")
    plt.legend()
    plt.savefig("plot.png")
