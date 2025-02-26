import numpy as np
import numpy.linalg as la

n = 5
X = np.random.randint(-1000, 1000, (n, n))
A = X@X.T + np.eye(n)
b = np.random.randint(-1000, 1000, (n, 1))
x = np.random.randint(-1000, 1000, (n, 1))

def G(x, A, b):
    return A@x + b

def solve(x, A, G):
    V = la.eig(A)[1]
    i = 0
    while la.norm(G(x, A, b)) >= 1e-7 :
        if i >= len(A):
            break
        d = V[:, i].reshape(-1, 1)
        alpha = - d.T @ G(x, A, b) / ( d.T @ A @ d )
        x = x + alpha*d
        i += 1
        print(la.norm(G(x, A, b)))
    return x

def solveNonQuad(x, G, A):
    k = 0
    while la.norm(G(x, A, b)) >= 1e-7:
        d = -G(x, A, b)
        alpha = - d.T @ G(x, A, b) / ( d.T @ A @ d )
        x = x + alpha*d
        if k%n != 0:
            d = -G(x, A, b) + (( G(x, A, b).T @ (-d) )/( d.T @ d ))
        else:
            d = -G(x, A, b)
        k += 1
    return x
print(np.allclose(solveNonQuad(x, G, A), -la.inv(A) @ b))