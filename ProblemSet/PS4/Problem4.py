'''
Consider that you are given a set of vectors a1, a2, . . . , al with ai ∈Rm,
and another set of vectors b1, b2, . . . , bl with bi ∈Rn. Write a code to
construct a matrix A such that Col(A) = span(a1, . . . , al) and Row(A) =
span(b1, b2, . . . , bl). Your code must print “Not possible” if such a matrix
cannot be constructed.
'''
import numpy as np
import numpy.linalg as la
def ortho_basis(X):

    return la.qr(X)[0][:, :la.matrix_rank(X)], la.matrix_rank(X)

A = eval(input("Enter list of vectors in A"))
B = eval(input("Enter list of vectors in B"))

U, ur = ortho_basis(A)
V, vr = ortho_basis(B)

if ur != vr:
    print("Not Possible")
else:
    print(U@V.T)




