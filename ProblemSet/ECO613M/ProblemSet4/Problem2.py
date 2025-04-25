import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model as lm
import sklearn.metrics
np.random.seed(42)

X1 = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], 10)
X2 = np.random.multivariate_normal([0, 1], [[1, 0], [0, 1]], 10)

XX1 = np.vstack([np.random.multivariate_normal(X1[k], [[0.1, 0], [0, 0.1]], 10) for k in range(10)])
XX2 = np.vstack([np.random.multivariate_normal(X2[k], [[0.1, 0], [0, 0.1]], 10) for k in range(10)])
X = np.vstack([XX1, XX2])

Y1 = np.ones(100)
Y2 = -np.ones(100)
Y = np.concatenate([Y1, Y2])

clf = lm.LogisticRegression().fit(X, Y)
x_vals = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
y_vals = -(clf.coef_[0, 0] * x_vals + clf.intercept_[0]) / clf.coef_[0, 1]

predictions = clf.predict(X)
error = sklearn.metrics.zero_one_loss(predictions, Y)
print(error)
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color = 'red', label = 'Class + 1')
plt.scatter(X[Y == -1][:, 0], X[Y == 1][:, -1], color = 'yellow', label = 'Class - 1')
plt.plot(x_vals, y_vals)
plt.show()


