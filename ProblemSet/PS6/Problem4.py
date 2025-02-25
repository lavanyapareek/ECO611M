'''
Let f (x) = x1 * exp(-x1^2 - x2^2). Solve for its minimum using gradient descent with
backtracking line search. Let x0 = (1, 1). Print the number of iterations
needed for convergence to the solution (-1/√2 , 0)
'''
import numpy as np
import numpy.linalg as la
def G(x):
    return np.array([
        (1 - 2*x[0]**2) * np.exp(-x[0]**2 - x[1]**2),
        -2*x[1] * x[0] * np.exp(-x[0]**2 - x[1]**2)
        ])
def btls(d, x):
    alpha = 1
    c1 = 0.8
    rho = 0.8
    while F(x + alpha*d) - F(x) > -c1*(np.dot(G(x), d)):
        alpha = rho*alpha
    return alpha
def F(x):
    return -x[0]*np.exp(-x[0]**2 - x[1]**2)
def solve(x):
    k = 0
    while la.norm(G(x)) >= 1e-7 :
        d = -G(x)
        alpha = btls(d, x)
        x = x + alpha*d
        k = k + 1
    return x, k

print(solve(np.array([0.7, 1])))
