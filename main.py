import math

import numpy as np
import sympy as sy

N = 87463


def primfacs(n):
    i = 2
    primfac = []
    while i * i <= n:
        while n % i == 0:
            primfac.append(i)
            n = n / i
        i = i + 1
    if n > 1:
        primfac.append(n)
    return primfac


def pow_two(p):
    res = []
    n = p - 1
    S = 0
    prom = n
    while prom % 2 == 0:
        prom /= 2
        S += 1
    res.append(S)
    Q = math.ceil(n / math.pow(2, S))
    res.append(Q)

    return res


def tonelli_shanks(n, p):
    if not eulers_criteria(n, p):
        return 0
    QS_arr = pow_two(p)
    n = n % p
    S = QS_arr[0]
    Q = QS_arr[1]

    z = 2
    while True:
        if eulers_criteria(z, p):
            z += 1
        else:
            break
    M = S % p
    c = int(math.pow(z, Q)) % p
    t = int(math.pow(n, Q)) % p
    R = int(math.pow(n, int((Q + 1) / 2))) % p

    while t != 1:
        i = 0
        for j in range(1, M):
            if int(math.pow(t, math.pow(2, j))) % p == 1:
                i = j
                break
        b = int(math.pow(c, math.pow(2, M - i - 1))) % p
        M = i % p
        c = int(math.pow(b, 2)) % p
        t = t * c % p
        R = R * b % p

    return [R, (R * (-1)) % p]


def eulers_criteria(n, p):
    res = 1
    n = n % p
    pow = math.ceil((p - 1) / 2)

    while pow:
        if pow & 1:
            res = (res * n) % p
        n = (n * n) % p
        pow = pow >> 1

    if res == 1:
        return True
    return False


def F_a(num):
    return math.pow(num, 2) - N  # за модуль не выйдешь
    pass


def build_factor_base(num):  # Это дерьмо можно распарралелить
    a = [i for i in range(0, num + 1)]
    a[1] = 0
    lst = []
    i = 2
    while i <= num:
        if a[i] != 0:
            lst.append(a[i])
            for j in range(i, num + 1, i):
                a[j] = 0
        i += 1
    return lst


def sieve_numb():

    m = 296

    A = math.ceil(pow(math.exp(math.sqrt(math.log(N) * math.log(math.log(N)))), 1 / math.sqrt(2))) + 1
    LS = []
    for i in range(0, 52):
        LS.append(F_a(m + i))
    FB_prom = build_factor_base(A)
    FB = []
    for i in FB_prom:
        if i == 2:
            FB.append(i)
            continue
        if eulers_criteria(N, i):
            FB.append(i)

    matrix = sy.zeros(len(LS), len(FB))

    for i in range(len(FB)):
        prom = tonelli_shanks(N, FB[i])
        index_one = (prom[0] - m) % FB[i]
        index_two = (prom[1] - m) % FB[i]

        for j in range(0, len(LS)):
            if index_one == j:
                while LS[j] % FB[i] == 0:
                    LS[j] = LS[j] / FB[i]
                    matrix[j, i] += 1
                index_one += FB[i]
            if index_two == j:
                while LS[j] % FB[i] == 0:
                    LS[j] = LS[j] / FB[i]
                    matrix[j, i] += 1
                index_two += FB[i]

    print(matrix)
    prom = []
    for i in range(0, len(LS)):
        if LS[i] == 1:
            prom.append(matrix.row(i))
            # print(i)

    # A = matrix(R,prom)

    print(prom)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # print(pow_two(13))
    # sieve_numb()
    # matr = sy.Matrix([[0,0,0,1,0,0,0,0,0],
    #                   [1,1,0,1,1,0,0,0,0],
    #                   [0,1,0,0,0,0,1,0,1],
    #                   [1,0,1,0,0,0,1,0,0],
    #                   [0,0,0,1,0,0,0,0,0],
    #                   [1,1,1,0,0,0,0,0,1],
    #                   [1,1,0,0,1,0,0,0,0]])
    # print(matr.inv_mod(2))
    matrix = np.matrix(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
        ]
    )
    matrix = matrix.transpose()
    print(np.triu(matrix))
    # print(np.triu(    [ [0,0,0,1,0,0,0,0,0],
    #                     [1,1,0,1,1,0,0,0,0],
    #                     [0,1,0,0,0,0,1,0,1],
    #                     [1,0,1,0,0,0,1,0,0],
    #                     [0,0,0,1,0,0,0,0,0],
    #                     [1,1,1,0,0,0,0,0,1],
    #                     [1,1,0,0,1,0,0,0,0]] ))
    # print(matr)
    #
    # check = matr.inv_mod(2)
    # print(check)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
