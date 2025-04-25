import numpy as np
import scipy.stats as sps
np.random.seed(42)
def nCr(n, r):
    if r > n:
        return 0
    r = min(r, n - r)
    result = 1
    for i in range(1, r + 1):
        result *= n - r + i
        result //= i
    return result
def generate_binom(N, p, n):
    U = np.random.rand(N, n)
    return (U < p).sum(axis = 1)

def PMFBinom(k, n, p):
    return nCr(n, k) * (p ** k) * ((1 - p) ** (n - k))

def binom_test(obs, p, n):
    return 0

def find_p_binom(obs_binom, n):
    P = np.arange(0, 1, 0.001)
    maxP = -1
    res = -1
    for p in P:
        curr_p = sps.binomtest(obs_binom.sum(), n*len(obs_binom), p).pvalue
        if curr_p > maxP:
            maxP = curr_p
            res = p
    return res
N = 10000
p = 0.111
n = 5
obs_binom = generate_binom(N, p, n)


#print(binom_test(obs_binom, p, n))
print(find_p_binom(obs_binom, n))
print(sps.binomtest(obs_binom.sum(), n*len(obs_binom), 0.112).pvalue)



