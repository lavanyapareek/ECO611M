import numpy as np
import numpy.linalg as la

def G(x):
    return np.array([200*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)])
def F(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
def btls(x, d):
    alpha = 1
    rho = 0.8
    c1 = 0.3
    while F( x + alpha*d ) - F(x) > c1*alpha* d.T @ G(x) :
        alpha = rho*alpha
    return alpha

def solve(x, n):
    d = -G(x)
    k = 0
    while la.norm(G(x)) >= 1e-12:
        alpha = btls(x, d)
        x_old = x
        x = x + alpha*d
        if k%n != 0:
            d = -G(x) + (G(x).T @ (G(x))) / (G(x_old).T @ G(x_old))*d
        else :
            d = -G(x)
        k += 1
    return x, k
print(solve(np.array([10,10]), 2))