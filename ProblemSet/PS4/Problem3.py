'''
Assume that you are given a matrix A = [a1, a2, . . . , am] where ai ∈Rn are
n-length vectors. Consider m < n, and that A is an orthonormal matrix.
Write a code to construct another (n-m) vectors am+1, am+2, . . . , an such
that the matrix B = [a1, . . . , an] is still an orthonormal matrix.
'''
import numpy as np
import numpy.linalg as la

def ortho(A):
    m, n = A.shape
    if n >= m:
        return ValueError("m should not be greater than n")

    return np.hstack((A, la.qr(A, mode="complete")[0][:, n :]))

def orthosvd(A):
    m, n = A.shape
    if n >= m:
        return ValueError("m should not be greater than n")

    return np.hstack((A, la.svd(A)[0][:, n :]))

A = np.random.rand(3, 2)
A, _ = la.qr(A)

print(A)
print(ortho(A))
print(orthosvd(A).T@orthosvd(A))