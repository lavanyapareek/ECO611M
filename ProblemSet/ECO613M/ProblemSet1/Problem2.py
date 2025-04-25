import numpy as np
import matplotlib.pyplot as plt

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

plt.hist(poisson(1000), bins = 500)
plt.show()