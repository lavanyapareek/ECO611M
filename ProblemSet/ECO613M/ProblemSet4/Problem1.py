import numpy as np
import matplotlib.pyplot as plt

X1 = np.random.multivariate_normal([1, 0], [[1,0], [0,1]], 10)
X2 = np.random.multivariate_normal([0, 1], [[1,0], [0,1]], 10)

XX1 = np.vstack([np.random.multivariate_normal(X1[i], [[0.1, 0], [0, 0.1]], 10) for i in range(10)])
XX2 = np.vstack([np.random.multivariate_normal(X2[i], [[0.1, 0], [0, 0.1]], 10) for i in range(10)])
X = np.vstack([XX1, XX2])

Y1 = np.ones(100)
Y2 = -np.ones(100)
Y = np.concatenate([Y1, Y2])


plt.scatter(X[Y == 1][: , 0], X[Y == 1][:, 1], color = 'yellow', label = 'Class +1')
plt.scatter(X[Y == -1][: , 0], X[Y == -1][:, 1], color = 'red', label = 'Class -1')
plt.grid(True)
plt.show()


