import numpy as np
import scipy.stats as sps
def nCr(n, r):
    if r > n:
        return 0
    r = min(r, n - r)
    result = 1
    for i in range(1, r + 1):
        result *= n - r + i
        result //= i
    return result

def probab(a, b, c, d):
    return (nCr(a + b, a) * nCr(c + d, d)) / nCr(a + b + c + d, a + c)
a = 1000
b = 1000
c = 1000
d = 1000
p_val = 0
print(sps.fisher_exact([[a, b], [c, d]], 'greater'))
while c >= 0 and b >= 0:
    p_val += probab(a, b, c, d)
    a += 1
    c -= 1
    b -= 1
    d += 1
print(p_val)

