%%time

from pprint import pprint

import numpy as np

from itertools import compress
from functools import reduce
import random
import time
PRIMES_FILE = 'primes.txt'


def f(t: int, n: int) -> int:
    return int(t ** 2 - n)


# noinspection PyPep8Naming
def get_FB(b: int):
    with open(PRIMES_FILE) as file:
        primes = list(filter(lambda x: x <= b, map(int, file.read().split(','))))
        return primes


# noinspection PyPep8Naming
def get_LS(a: int, b: int, n: int):
    return [int(f(a + i, n)) for i in range(b + 1)]


def euler_criteria(n: int, p: int):
    return int(pow(n, (p - 1) // 2, p)) == 1


# noinspection PyPep8Naming
def filter_FB(n: int, fb):
    return list(filter(lambda x: euler_criteria(n, x), fb))


def get_vectors(ls, fb):
    res = []
    indexes = []
    for idx, ls_el in enumerate(ls):
        row = []
        for fb_el in fb:
            el_pow = 0
            while ls_el % fb_el == 0:
                ls_el //= fb_el
                el_pow += 1
            row.append(el_pow)
        if ls_el == 1:
            indexes.append(idx)
            res.append(row)
    return indexes, res

def main(N, need_print=True):
    const_a = int(np.ceil(np.sqrt(N)))
    L = np.exp(np.sqrt(np.log(N) * np.log(np.log(N))))
    B = int(np.ceil(np.power(L, 1 / np.sqrt(2)))) * 110
    FB = get_FB(B)
    LS = get_LS(const_a, B, N)
    FB = filter_FB(N, FB)
    indexes, vectors = get_vectors(LS, FB)
    a_list = [i + const_a for i in indexes]
    ls_list = [LS[i] for i in indexes]
    R = Zmod(2)
    A = Matrix(R, vectors)

    ker = A.left_kernel()


    for row in ker.basis():
        x = reduce(lambda x,y: x * y % N, compress(ls_list, row))
        a_gcd = int(gcd(x - sqrt(x), N))
        if a_gcd != 1:
            if need_print:
                print(f"n = {N} = {a_gcd} * {N // a_gcd}")
            break
    else:
        if need_print:
            print(f"n = {N} is prime")

print("Compile ...")
main(10_127, False)

print("RunTime")
res_n = []
res_t = []
# for cur_pow in range(3, 10 + 1):
#     cur_n = 10 ** cur_pow + random.randint(123, 321)
for cur_n in [1251,10173,100247,1000225,10000188,100000185,1000000179,10000000177]:
    t = time.perf_counter()
    main(cur_n)
    c_time = time.perf_counter() - t
    res_n.append(cur_n)
    res_t.append(c_time)

plt.title("Факторизация - последовательно")
plt.xlabel("Numbers")
plt.ylabel("Times, s")
plt.xscale('log')
plt.plot(res_n, res_t, color="red")
plt.show()
