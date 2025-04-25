import numpy as np
import numpy.linalg as la
n = 5
X = np.random.randint(-9, 10, (n, n))
A = X.T @ X + np.eye(n)
b = np.random.randint(-10, 10, (n, 1))
x = np.random.randint(-10, 10, (n, 1))

V = la.eig(A)[1]

i = 0
while la.norm(A @ x + b) >= 1e-6:
    d = V[:, i].reshape(-1, 1)
    alpha = - (d.T @ (A @ x + b) ) / (d.T @ A @ d)
    x = x + alpha * d
    i += 1
print(x, la.norm(A @ x + b))
print(np.allclose(x, -la.inv( A ) @ b))
def F(x, A):
    return 0.5*x.T @ A @ x + b.T @ x
def btls(x, d, g, A):
    alpha = 1
    rho = 0.8
    c1 = 0.01
    while F(x + alpha * d, A) - F(x, A) > c1 * alpha * d.T @ g:
        alpha = rho * alpha 
    return alpha
def G(x, A):
    return A @ x + b

x = 0*np.random.randint(-10, 10, (n, 1))
d = -G(x, A)
k = 0

while la.norm(A @ x + b) >= 1e-6:
    alpha = btls(x, d, G(x, A), A)
    x_old = x
    x = x + alpha * d
    if k%n != 0:
        d = -G(x, A) + ((G(x,A).T @ G(x, A)) / ((G(x_old,A).T @ G(x_old, A))))*d
    else:
        d = -G(x, A)
    k += 1
print(x, la.norm(A @ x + b), k)
print(np.allclose(x, -la.inv( A ) @ b))

