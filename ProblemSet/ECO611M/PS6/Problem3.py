'''
Let H= [[2 a], [a 2]] . Solve for the minimum of (1/2xTHx) for every a in the
set {-1.9,-1.8, . . . , 1.8, 1.9} using gradient descent method with exact
line search. Start with the same initial point for each value of a. Plot the
number of iterations for convergence as a function of a.
'''

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
A = np.arange(-1, 1, 0.001)
def F(x, a):
    return 0.5 * x.T @ H(a) @ x
def H(a):
    return [[15, a], [a, 15]]
def G(H, x):
    return np.dot(H, x)
def exactLineSearch(H, d):
    return (np.dot(d, d)/(d.T @ H @ d))
def btls(x, d, g, a):
    alpha = 1
    rho = 0.8
    c1 =0.1
    while F( x + alpha*d, a ) - F( x, a ) > c1 * alpha * d.dot(g):
        alpha = rho * alpha
    return alpha
def solve(H, x, a):
    k = 0
    while la.norm(G(H, x)) >= 1e-10:
        d = -G(H, x)
        alpha = btls(x, d, -d, a)
        x = x + alpha*d
        k = k + 1
    return x, k
x = np.array([1, 1])
K = []
for a in A:
    print(solve(H(a), x, a), a)
    K.append(solve(H(a), x, a)[1])
plt.plot(A, K)
plt.show()
