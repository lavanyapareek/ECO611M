import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def K_NN(X_train, Y_train, k, test=None):
    if test is None:
        test = X_train
    y_pred = []
    for i, x in enumerate(test):
        distances = np.linalg.norm(X_train - x, axis=1)
        NN = np.argsort(distances)[:k]
        NN_y = Y_train[NN].astype(int)
        pred = np.sign(np.sum(NN_y))
        y_pred.append(pred)
    return np.array(y_pred)

N = 10
X1 = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], N)
X2 = np.random.multivariate_normal([0, 1], [[1, 0], [0, 1]], N)

XX1 = np.vstack([np.random.multivariate_normal(X1[k], [[0.1, 0], [0, 0.1]], 10) for k in range(N)])
XX2 = np.vstack([np.random.multivariate_normal(X2[k], [[0.1, 0], [0, 0.1]], 10) for k in range(N)])
X = np.vstack([XX1, XX2])

Y1 = np.ones(10 * N)
Y2 = -np.ones(10 * N)
Y = np.concatenate([Y1, Y2])

k = 1
y_pred = K_NN(X, Y, k)
print(np.mean(y_pred != Y))


x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]


grid_pred = K_NN(X, Y, k)
Z_grid = K_NN(X, Y, k, grid).reshape(xx.shape)


plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_grid, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.grid(True)
plt.show()