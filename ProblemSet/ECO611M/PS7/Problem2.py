import numpy as np
import numpy.linalg as la

def F(x):
    return x[0] * np.exp( -x[0] ** 2 - x[1] ** 2 )

def G(x):
    f1 = ( 1 - 2 * x[0] ** 2) * np.exp( -x[0] ** 2 - x[1] ** 2 )
    f2 = -2 * x[1] * np.exp( -x[0] ** 2 - x[1] ** 2 )
    return np.array([f1, f2])
def btls(x, d):
    alpha = 1
    rho = 0.8
    c1 = 0.3
    while F(x + alpha*d) - F(x) > c1*alpha* d.T @ G(x):
        alpha = rho*alpha
    return alpha
def NonQuadSolver(x, n):
    k = 0
    d = -G(x)
    while la.norm(G(x)) >= 1e-8:
        alpha = btls(x, d)
        x = x + alpha*d
        if k%n != 0:
            d = -G(x) + ((G(x).T @ (G(x))) / (d.T @ d))*d
        else :
            d = -G(x)
        k += 1
        print(la.norm(G(x)))
    return x, k
x = np.array([-1, 0])
print(NonQuadSolver(x, 2))