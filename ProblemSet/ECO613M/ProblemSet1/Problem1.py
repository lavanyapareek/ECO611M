import numpy as np
import matplotlib.pyplot as plt
def compute_binomial(n,p):
    U = np.random.rand(1000000, n)
    samples = (U < p).sum(axis = 1)
    return samples
def compute_geometric(p):
    U = np.random.rand(1000000)
    samples = np.floor(np.log(1 - U) / np.log(1 - p)).astype(int) + 1
    return samples
binomial_samples = compute_binomial(100, 0.3)
geometric_samples = compute_geometric(0.5)
print(geometric_samples.mean())
print(geometric_samples.var())
print(binomial_samples.mean())
print(binomial_samples.var())
# plt.hist(binomial_samples)
# plt.show()
# plt.hist(geometric_samples)
# plt.show()