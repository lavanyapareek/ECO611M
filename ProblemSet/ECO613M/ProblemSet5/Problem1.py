import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.linear_model as sklm
import scipy
np.random.seed(42)

X = np.random.multivariate_normal(np.zeros(20), np.eye(20), 100)
i1, i2, i3, i4 = np.random.randint(0, 20, 4)
a, b, c, d = 0.5*np.random.randn(4)
epsilon = 0.1*np.random.randn(100)
Y = a * X[:, i1] + b * X[:, i2] + c * X[:, i3] + d * X[:, i4] + epsilon

print(i1, i2, i3, i4)

# X1 = np.hstack([np.ones((X.shape[0], 1)), X])

# beta_chapeau = np.linalg.inv(X1.T @ X1) @ X1.T @ Y

# model = sklm.LinearRegression(fit_intercept=True)
# model.fit(X1, Y)

# beta_chapeau_sklearn = model.coef_

# print("\nDifference between closed-form and sklearn:\n", beta_chapeau - beta_chapeau_sklearn)
# print(np.sum(np.isclose(beta_chapeau, beta_chapeau_sklearn)))

# alpha = 20
# beta_chapeau_ridge = np.linalg.inv(X.T @ X + alpha * np.eye(20)) @ X.T @ Y
# beta0_ridge = np.mean(Y) - np.sum(np.mean(X, axis = 0) * beta_chapeau_ridge)
# beta_chapeau_ridge = np.concatenate((beta_chapeau_ridge.ravel(), [beta0_ridge]))
# print(beta_chapeau_ridge)

# model = sklm.Ridge(alpha = alpha, fit_intercept=True)
# model.fit(X, Y)
# beta_chapeau_ridge_sklearn = np.concatenate((model.coef_, [model.intercept_]))

# print(np.argsort(np.abs(beta_chapeau_ridge))[-5:], np.argsort(np.abs(beta_chapeau_ridge_sklearn))[-5:])
def lasso(beta, X, Y, alpha):
    return np.sum(Y - X @ beta) + alpha * np.linalg.norm(beta)
alpha = 20
beta_chapeau_lasso = scipy.optimize.minimize()








