import numpy as np
import matplotlib.pyplot as plt

X = np.zeros(100000)

for i in range(100000):
    u1 = np.random.rand()
    u2 = np.random.rand()
    X[i] = -np.log((1-u1)*(1-u2))

plt.hist(X, bins=np.arange(0, 25, 0.1), density=True)
plt.show()

