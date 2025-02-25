'''
Let H= [[2 a], [a 2]] . Solve for the minimum of (1/2xTHx) for every a in the
set {-1.9,-1.8, . . . , 1.8, 1.9} using gradient descent method with exact
line search. Start with the same initial point for each value of a. Plot the
number of iterations for convergence as a function of a.
'''

import numpy as np
import numpy.linalg as la

A = np.arange(-1.9, 2, 0.1)
def H(a):
    return [[2, a], [a, 2]]
def G(H, x):
    return np.dot(H, x)
def exactLineSearch(H, d):
    return (np.dot(d, d)/(d.T @ H @ d))
def solve(H, x):
    k = 0
    while la.norm(G(H, x)) >= 1e-10:
        d = -G(H, x)
        alpha = exactLineSearch(H, d)
        x = x + alpha*d
        k = k + 1
    return x, k
x = np.array([100, 1])
for a in A:
    print(solve(H(a), x), a)

