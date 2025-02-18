'''
Consider the given matrix A of size mxn.
(a) Write a code to split the given vector x ∈Rm as x = xc + xln, where xc ∈ Col(A) and xln ∈ Null(AT ).
(b) Write a code to split the given vector x ∈Rn as x = xr + xn, where xr ∈ Row(A) and xn ∈ Null(A).
(Hint: Use projection!)
'''
import numpy.linalg as la
import numpy as np
def split_col_lnull(A, x):
    P_A = A @ la.pinv(A)
    print(P_A)
    xc = P_A @ x
    xln = x - xc
    return xc, xln
def split_row_null(A, x):
    A = A.T
    P_A = A @ la.inv(A.T @ A) @ A.T
    xr = P_A @ x
    xn = x - xr
    return xr, xn

A = np.array([[1,2,3],[4,5,6],[13,17,19]])

print(split_col_lnull(A, np.array([[1,2,3], [5,6,7]])))
print(split_row_null(A, np.array([4,5,6])))
