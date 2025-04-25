'''
Multiply Two Matrices
'''
import numpy as np
X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ])
Y = np.array([
    [1,1],
    [1,1],
    [1,1]
    ])
#print(np.matmul(X,Y))

N = len(X)
M = len(Y)
P = len(Y[0])

res = np.full((N,P), 0)

for i in range(N):
    for j in range(P):
        for k in range(M):
            res[i][j] += X[i][k]*Y[k][j]
print(res)