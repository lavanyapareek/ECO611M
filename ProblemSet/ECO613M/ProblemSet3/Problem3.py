import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
data = genfromtxt('/Users/lavanyapareek/Documents/Python and ML/ProblemSet/ECO613M/ProblemSet1/ProblemSet3/bjm.csv', delimiter= ',')
X = np.array(data[:,0])

X_centered = X - X.mean()
p = 0

def get_autocorrelation(X, p):
    X_centered = X - np.mean(X)
    n = len(X)
    var = np.sum(X_centered ** 2) / n
    autocorrs = []
    for k in range(p + 1):
        cov = np.sum(X_centered[:n - k] * X_centered[k:]) / (n - k)
        autocorrs.append(cov / var)
    return np.array(autocorrs)

p_max = 100
gammas = get_autocorrelation(X_centered, p_max)
GAMMAS = np.array([[gammas[abs(i - j)] for j in range(p_max)] for i in range(p_max)])


def getAIC(sigma2, p, n):
    return np.log(np.abs(sigma2)) + 2 * p ** 2 / n


AIC_vals = []

for p in range(p_max):
    gamma_vec = gammas[1:p+1].reshape(-1, 1)
    G = GAMMAS[: p , : p]
    A = np.linalg.solve(G, gamma_vec)
    AIC_vals.append(getAIC(gammas[0] - np.dot(A.T, gamma_vec), p, len(X)))
p = np.argmin(np.array(AIC_vals))

print(p)
gamma_vec = gammas[1:p+1].reshape(-1, 1)
G = GAMMAS[: p , : p]
A = np.linalg.solve(G, gamma_vec)
sigma2 = gammas[0] - np.dot(A.T, gamma_vec)
mu = X.mean() * (1 - np.sum(A))
print(mu, sigma2)

