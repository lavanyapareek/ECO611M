import numpy as np
import scipy
from math import factorial
np.random.seed(42)

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

observed = poisson(10)
unique, count = np.unique(observed, return_counts = True)
obs_freq = dict(zip(unique, count))
exp = []
obs = []

k = max(unique)

for i in range(k + 1):
    Oi = obs_freq.get(i, 0)
    Ei = len(observed) * pmfPois(i, 10)

    exp.append(Ei)
    obs.append(Oi)

total_obs = sum(obs)
total_exp = sum(exp)
exp = [e * total_obs / total_exp for e in exp]

print(scipy.stats.chisquare(f_obs = obs, f_exp = exp))

k = max(unique) + 1
stat = 0
dof = 0
for i in range(k):
    Oi = obs_freq.get(i, 0)
    Ei = len(observed) * pmfPois(i, 10)

    if Oi > 0 and Ei < 1e-10 :
        continue
    if Oi > 0:
        dof += 1
        stat += (( Oi - Ei ) ** 2 )/ Ei

print(1 - scipy.stats.chi2.cdf(stat, dof))
