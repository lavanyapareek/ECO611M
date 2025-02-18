'''
Consider the Cournot's oligopoly setting with three firms. Compute the
quantities produced at equilibrium if
(a) P (X) = max(0, 1-X), Ci(x) = x log(x).
(b) P (X) = max(0, 1-X^2), Ci(x) =x
2.
(c) P (X) = max(0, 1-X), Ci(x) = xi
.
'''
import numpy as np
import numpy.linalg as la
def Jacob(c, x):
    if c == 'A':
        return np.array([
            [-2 - 1/x[0], -1, -1],
            [-1, -2 - 1/x[1], -1],
            [-1, -1, -2 - 1/x[2]]
        ])
    elif c == 'B':
        return np.array([
            [-2*(x[0] + x[1] + x[2]) - 4*x[0], -2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2])],
            [-2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2]) - 4*x[1], -2*(x[0] + x[1] + x[2])],
            [-2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2]), -2*(x[0] + x[1] + x[2]) - 4*x[2]]
        ])
    else:
        return np.array([
            [-2, -1, -1],
            [-1, -4, -1],
            [-1, -1, -2 -6*x[2]]
        ])
def solve(f, x, c):
    k = 0
    while f(x)[0]**2 + f(x)[1]**2 + f(x)[2]**2 >= 1e-6:
        x = x - la.pinv(Jacob(c, x))@np.array([f(x)[0], f(x)[1], f(x)[2]])
        k += 1
        print(f(x)[0]**2 + f(x)[1]**2 + f(x)[2]**2, x, k)
        if k > 99:
            break
    return x
def fA(x):
    return [1 - x[0] - x[1] - x[2] - x[0] - np.log(x[0]) -  1, 1 - x[0] - x[1] - x[2] - x[1] - np.log(x[1]) -  1, 1 - x[0] - x[1] - x[2] - x[2] - np.log(x[2]) -  1]
def fB(x):
    return [1 - (x[0] + x[1] + x[2])**2 - 2*x[0]**2 - 1/2, 1 - (x[0] + x[1] + x[2])**2 - 2*x[1]**2 - 1/2, 1 - (x[0] + x[1] + x[2])**2 - 2*x[2]**2 - 1/2]
def fC(x):
    return [1 - x[0] - x[1] - x[2] - x[0] - 1, 1 - x[0] - x[1] - x[2] - x[1] - 2*x[1], 1 - x[0] - x[1] - x[2] - x[2] - 3*x[2]**2]
print(solve(fA, np.array([0.5, 0.5, 0.5]),  'A'))
print(solve(fB, np.array([0.5, 0.5, 0.5]),  'B'))
print(solve(fC, np.array([0.5, 0.5, 0.5]),  'C'))
