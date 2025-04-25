import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
np.random.seed(42)
def gamma(alpha, beta, N):
    gamma = np.zeros(N)
    for i in range(alpha):
        gamma += np.random.exponential(1/beta, N)
    return gamma

def empirical_cdf(data, x):
    return np.mean(data <= x)

def KolmSmirn(data, alpha, beta, N):
    DnMax = 0
    for x in data:
        F_emp = empirical_cdf(data, x)
        F_th = sps.gamma.cdf(x, alpha, scale = 1 / beta)
        Dn = abs(F_emp - F_th)
        if Dn > DnMax:
            DnMax = Dn
    return DnMax

def KolmCdf(x, N = 100):
    v = 0
    for i in range(1, N):
        v += np.exp(-((2*i - 1) ** 2) * (np.pi ** 2) ) / ( 2 * x ** 2 )
    return (((2 * np.pi) ** (0.5)) / x) * v

alpha = 3
beta = 4
N = 10000
G = gamma(alpha, beta, N)
Dn = KolmSmirn(G, alpha, beta, N)

print(Dn)
print(sps.kstest(G, 'gamma', args = (alpha, 0, 1/beta)))
print(1 - KolmCdf(np.sqrt(N)*Dn))




