'''
Consider that you are given a set of vectors a1, a2, . . . , al with ai ∈Rm,
and another set of vectors b1, b2, . . . , bl with bi ∈Rn. Write a code to
construct a matrix A such that Col(A) = span(a1, . . . , al) and Row(A) =
span(b1, b2, . . . , bl). Your code must print “Not possible” if such a matrix
cannot be constructed.
'''
import numpy as np
import numpy.linalg as la
#A = eval(input("Enter list of vectors in A"))
#B = eval(input("Enter list of vectors in B"))
A = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
B = np.array([[1,0],[0,1]])
print((la.qr(A)[0][:, :la.matrix_rank(A)]) @ (la.qr(B.T)[0][:, :la.matrix_rank(B)].T) if la.matrix_rank(A) == la.matrix_rank(B) else "Not Possible")

