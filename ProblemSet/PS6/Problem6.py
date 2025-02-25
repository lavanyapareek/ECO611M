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

n = 1000
X = np.random.randint(-10,10,(n,n))
A = X@X.T + np.eye(n)
b = np.random.randint(-100, 100, (n,1))
L, V = la.eig(A)
x = np.zeros(n)
g = A@x + b
i = 0
while la.norm(g) >= 1e-6:
    if i >= n:
        break
    d = V[:, i]
    alpha = -(d.T @ g)/(d.T @ A @ d)
    x = x + alpha*d
    g = A@x + b.flatten()
    i += 1
    print(la.norm(A@x + b))

print(np.allclose(x.reshape(-1, 1), -la.inv(A)@b))