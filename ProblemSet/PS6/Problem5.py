'''
Solve for the minimum of the Rosenbrock function 100(x2-x1^2)^2 - (1 - x1)^2
using gradient descent algorithm with backtracking line search. Start
with (0.5, 0.5). Print the number of iterations needed for converging to
the solution (1, 1)
'''
import numpy as np
import numpy.linalg as la

def G(x):
    return np.array([200*(x[1] - x[0]**2)*(-2*x[0]) - 2*(1 - x[0]), 200*(x[1] - x[0]**2)])
def F(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
x = np.array([1/2, 1/2])
k = 0
while la.norm(G(x)) >= 1e-6:
    print(la.norm(G(x)))
    d = -G(x)
    alpha = 1
    c1 = 0.5
    rho = 0.1
    while(F(x + alpha*d) - F(x) > c1*alpha*np.dot(d, -d)):
        alpha = rho*alpha
    x = x + alpha*d
    k += 1
print(x, k)
