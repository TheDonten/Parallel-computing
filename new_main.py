import math
import random
import time
import warnings
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from numpy import typing as npt

from utils import calc_ls, filter_fb, cals_vectors, left_null_space

warnings.filterwarnings("ignore")
threads_per_block = 1024

PRIMES_FILE = "primes.txt"


def get_ls(a, b, n) -> npt.NDArray[int]:
    LS_np = np.zeros(b, dtype=np.int64)

    blocks_per_grid = (LS_np.size + (threads_per_block - 1)) // threads_per_block
    res = cuda.device_array_like(LS_np)

    calc_ls[blocks_per_grid, threads_per_block](a, n, res)

    return res.copy_to_host()


def get_fb(b: int) -> npt.NDArray[int]:
    with open(PRIMES_FILE) as file:
        primes = list(filter(lambda x: x <= b, map(int, file.read().split(','))))
        return np.array(primes, dtype=np.int64)


def get_filtered_fb(n: int, fb: npt.NDArray[int]) -> npt.NDArray[int]:
    blocks_per_grid = (fb.size + (threads_per_block - 1)) // threads_per_block
    res = cuda.device_array(fb.shape, dtype=np.int64)
    filter_fb[blocks_per_grid, threads_per_block](n, cuda.to_device(fb), res)

    res = res.copy_to_host()
    return res[res > 0]


def get_vectors(ls: npt.NDArray[int], fb: npt.NDArray[int]) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
    res = cuda.device_array_like(np.zeros((ls.size, fb.size), dtype=np.int64))
    indexes = cuda.device_array_like(np.zeros(ls.shape, dtype=np.int64))

    blocks_per_grid = (ls.size + (threads_per_block - 1)) // threads_per_block
    cals_vectors[blocks_per_grid, threads_per_block](cuda.to_device(ls), cuda.to_device(fb), indexes, res)

    res_indexes = indexes.copy_to_host()
    res_res = res.copy_to_host()
    res_indexes = res_indexes[res_indexes != -1]

    return res_indexes, res_res[res_indexes]


def main(n, need_print=True):
    const_a = int(np.ceil(np.sqrt(n)))
    L = np.exp(np.sqrt(np.log(n) * np.log(np.log(n))))
    B = int(np.ceil(np.power(L, 1 / np.sqrt(2)))) * 50
    FB = get_fb(B)
    LS = get_ls(const_a, B, n)
    FB = get_filtered_fb(n, FB)
    indexes, vectors = get_vectors(LS, FB)
    vectors = vectors % 2
    ker = left_null_space(vectors)
    ls_slice = LS[indexes]
    for row in ker:
        x = reduce(lambda x, y: x * y % n, np.compress(row, ls_slice))
        a_gcd = int(math.gcd(x - int(math.sqrt(x)), n))
        if a_gcd != 1:
            if need_print:
                print(f"n = {n} = {a_gcd} * {n // a_gcd}")
            break
    else:
        if need_print:
            print(f"n = {n} is prime")


if __name__ == '__main__':
    # N = 10 ** 12 + 157
    print("Compile ...")
    main(10_127, False)

    print("RunTime")
    res_n = []
    res_t = []
    for cur_pow in range(3, 10 + 1):
        cur_n = 10 ** cur_pow + random.randint(123, 321)
        t = time.perf_counter()
        main(cur_n)
        c_time = time.perf_counter() - t
        res_n.append(cur_n)
        res_t.append(c_time)

    plt.title("Факторизация - параллельно")
    plt.xlabel("Numbers")
    plt.ylabel("Times, s")
    plt.xscale('log')
    plt.plot(res_n, res_t, color="red")
    plt.show()
