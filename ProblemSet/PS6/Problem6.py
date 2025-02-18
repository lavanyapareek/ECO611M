'''
Generate a 5x5 matrix X where each element is a random integer in
the range {-9, . . . , 9}. Generate a positive definite matrix A as A=XXT + I5. 
Also generate a 5-length vector b. Now solve for the minimum of (1/2xTAx + bTx) as follows: 
(i) Find the eigenvectors of A. Let them be v0, v1, v2, v3, v4; 
(ii) Set dk = vk for k = 0, 1, 2, 3, 4; 
(iii) Use exact line search in each iteration. 
Check if the solution converges to the solution (-A^-1b) for all the generated (A, b).
'''
import numpy as np
import numpy.linalg as la

def generateRandomMatrix(n):
    return np.random.randint(-9, 10, (n, n))

def A(n):
    x = generateRandomMatrix(n)
    return x@x.T + np.eye(n)
def b(n):
    return np.random.randint(-9, 10, (n, 1))
n = 5
A = A(n)
b = b(n)
V, L = la.eig(A)

x = np.ones((n, 1))
g = A@x + b
d = -g
tol = 1e-10
k = 0

while la.norm(A@x + b) >= tol:
    alpha = -(d.T @ g) / (d.T @ A @ d)
    x = x + alpha * d
    g_new = A @ x + b
    d = -g_new + ((g_new.T @ g_new)/(g.T @ g))*d
    g = g_new
    k += 1
    print(la.norm(A@x + b), x.T, k)
print(np.isclose(-la.inv(A)@b, x, rtol = tol))