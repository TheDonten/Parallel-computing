import time

import galois
import numpy as np

from numba import njit, cuda

BPG, TPG = 4096, 1024

@njit()
def f(t: int, n: int):
    return int(t ** 2 - n)


@cuda.jit()
def calc_ls(a: int, n: int, res: list[int]):
    i = cuda.grid(1)
    res[i] = f(a + i, n)


@njit()
def euler_criteria(n: int, p: int) -> bool:
    res = 1
    n = n % p
    c_pow = (p - 1) // 2

    while c_pow:
        if c_pow & 1:
            res = (res * n) % p
        n = (n * n) % p
        c_pow = c_pow >> 1

    if res == 1:
        return True
    return False


@cuda.jit()
def filter_fb(n: int, fb: list[int], res: list[int]):
    i = cuda.grid(1)
    cur = fb[i]
    is_qr = euler_criteria(n, cur)
    if is_qr:
        res[i] = cur
    else:
        res[i] = 0


@cuda.jit()
def cals_vectors(ls: list[int], fb: list[int], res_indexes: list[int], res_vectors: list[list[int]]):
    i = cuda.grid(1)
    if i >= ls.size:
        return
    ls_el = int(ls[i])
    for j in range(fb.shape[0]):
        fb_el = int(fb[j])
        el_pow = 0
        while ls_el % fb_el == 0:
            ls_el //= fb_el
            el_pow += 1
        res_vectors[i][j] = el_pow

    if ls_el == 1:
        res_indexes[i] = i
    else:
        res_indexes[i] = -1


@cuda.jit()
def outer(A, arr1, arr2):
    i = cuda.grid(1)
    if i < A.shape[0]:
        tmp = 0
        for k in range(A.shape[1]):
            A[i, k] = arr1[i] * arr2[k]


@cuda.jit()
def xor_matrix(A, B, C):
    i = cuda.grid(1)
    if i < C.shape[0]:
        for k in range(A.shape[1]):
            C[i, k] = A[i, k] ^ B[i, k]


def _row_reduce(matrix, n_cols):
    n_cols = matrix.shape[1] if n_cols is None else n_cols
    A_rre = matrix.copy()
    p = 0  # The pivot

    for j in range(n_cols):
        # Find a pivot in column `j` at or below row `p`
        idxes = np.nonzero(A_rre[p:, j])[0]
        if idxes.size == 0:
            continue
        i = p + idxes[0]  # Row with a pivot

        A_rre[[p, i], :] = A_rre[[i, p], :]

        idxes = np.nonzero(A_rre[:, j])[0]
        idxes = idxes[idxes != p]

        a1 = cuda.to_device(A_rre[idxes, j])
        b1 = cuda.to_device(A_rre[p, :])
        r = cuda.device_array_like(np.zeros((idxes.shape[0], A_rre.shape[1]), dtype=np.int64))

        outer[BPG, TPG](r, a1, b1)

        A_rre_copy = cuda.to_device(A_rre[idxes, :])
        xor_matrix[BPG, TPG](A_rre_copy, cuda.to_device(r), A_rre_copy)
        A_rre[idxes, :] = A_rre_copy.copy_to_host()

        p += 1
        if p == A_rre.shape[0]:
            break

    return A_rre, p


def left_null_space(mat):
    GF = galois.GF(2)
    m, n = mat.shape
    identity = np.array(GF(mat).Identity(m), dtype=mat.dtype)
    AI = np.concatenate((mat, identity), axis=-1)
    AI_rre, p = _row_reduce(AI, n)
    LN = AI_rre[p:, n:]
    LN_res, _ = _row_reduce(LN, None)
    return LN_res


@cuda.jit()
def _reduce_compress(n, ls, mat_row, res):
    i = cuda.grid(1)
    print(res[i])
    if i >= ls.size:
        return
    if mat_row[i] != 0:
        res[0] *= ls[i]
        res[0] %= n


# didnt work
def reduce_compress(n, ls, matrix_row):
    res = cuda.device_array_like(np.array([1], dtype=np.int64))
    _reduce_compress[BPG, TPG](n, cuda.to_device(ls), cuda.to_device(matrix_row), res)
    return int(res.copy_to_host()[0])
