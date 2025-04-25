'''
Write a user-defined function that implements the command numpy.reshape,
and verify if it gives the same answer as the command does.
'''
import numpy as np
#Random Matrix Generator :
n = int(10*np.random.rand() + 1)
m = int(10*np.random.rand() + 1)
n = 5
m = 4
print("Matrix of Size : ", n , m)
X = np.array([[(np.random.randint(0, 10)) for _ in range(m)] for _ in range (n)])
print(X)

def rererereshape(p, q, X):
    # Check if the total number of elements matches
    if p * q != len(X) * len(X[0]):
        return f"cannot reshape array of size {len(X) * len(X[0])} into shape {(p, q)}"
    res = np.zeros((p,q), dtype = int)  # Initialize result matrix
    for idx in range(m*n):
        res[idx//q][idx%q] = X[idx//len(X[0])][idx%len(X[0])]
    return res

#Using Reshape to verify user defined reshape function
print(rererereshape(2, 10, X))
print(X.reshape(2, 10))
print(np.array_equal(rererereshape(2, 10, X), X.reshape(2, 10)))

