# import numpy as np
# np.random.seed(42)

# def binomial(n, p):
#     U = np.random.rand(1000, n)
#     binom = (U < p).sum(axis = 1)
#     return binom

# def geom(p):
#     geom = []
#     for _ in range(1000):
#         count = 0
#         while np.random.rand() > p:
#             count += 1
#         geom.append(count)
#     return np.array(geom)
    
# def negativebinom(r, p):
#     negbinom = []
#     for _ in range(1000):
#         count = 0
#         succ = 0
#         failures = 0
#         while succ < r:
#             count += 1
#             if np.random.rand() < p:
#                 succ += 1
#             else:
#                 failures += 1
#         negbinom.append(failures)
#     return negbinom
# def poisso(l):
#     samples = np.zeros(10000)
#     for i in range(10000):
#         total_time = 0
#         count = 0
#         while total_time < 1:
#             total_time += np.random.exponential(1/l)
#             if total_time < 1:
#                 count += 1
#         samples [i] = count
#     return samples


# print(np.mean(binomial(1, 0.2)))
# print(np.mean(negativebinom(1, 0.2)))
# print(np.mean(geom(0.2)))


# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.stats as sps

# # Mixture sampler
# def multi_dist(p, n, mu, sig2):
#     U = np.random.rand(n)
#     dist = []
#     N = [np.sqrt(sig2[i]) * np.random.randn(n) + mu[i] for i in range(len(mu))]
#     for i in range(n):
#         if U[i] < p[0]:
#             dist.append(N[0][i])
#         else:
#             dist.append(N[1][i])
#     return np.array(dist)

# mu = [-1, 9]
# sig2 = [5, 10]
# p = [0.1, 0.9]
# n = 1000000
# X = multi_dist(p, n, mu, sig2)

# mu_cap = np.array(np.random.choice(X, 2))
# sig2_cap = np.array([X.var() + 1, X.var() + 1])
# pi_cap = np.array([0.5, 0.5])

# tol = 1e-6

# while True :
#     muO, sig2O, piO = mu_cap, sig2_cap, pi_cap
#     f = np.array([pi_cap[i] * sps.norm.pdf( X, mu_cap[i], np.sqrt(sig2_cap[i])) for i in range(2)])
#     fsum = f.sum(axis = 0)
#     gamma = f / fsum
#     pi_cap = gamma.sum(axis = 1) / n
#     mu_cap = np.array([np.sum(gamma[i] * X) / np.sum(gamma[i]) for i in range(2)])
#     sig2_cap = np.array([np.sum(gamma[i] * (X - mu_cap[i]) ** 2) / np.sum(gamma[i])for i in range(2)])

#     if np.linalg.norm(muO - mu_cap) < tol and np.linalg.norm(sig2O - sig2_cap) < tol :
#         break

# print(mu_cap, sig2_cap, pi_cap)
import numpy as np
import scipy
np.random.seed(42)

def factorial(x):
    if x == 0:
        return 1
    if x == 1:
        return 1
    return x * factorial(x - 1)
def pmfPois(x, l):
    return (l ** x) * np.exp(-l) / factorial(x)

def poisson(l):
    samples = np.zeros(10000, dtype = int)
    for i in range(10000):
        total_time = 0
        count = 0
        while total_time < 1:
            total_time += np.random.exponential(1/l)
            if total_time < 1:
                count += 1
        samples[i] = count
    return samples
X = poisson(10)

unique, count = np.unique(X, return_counts=True)
O = dict(zip(unique, count))

k = max(unique) + 1
stat = 0
dof = 0
for i in range(k):
    Oi = O.get(i, 0)
    Ei = len(X) * pmfPois(i, 10)
    if Oi > 0 and Ei > 0:
        stat += (Oi - Ei) ** 2 / Ei
        dof += 1
print(stat, 1 - scipy.stats.chi2.cdf(stat, dof))


