'''
Consider the system of linear equations of the form Ax= b. Given the
matrix A (of size mxn) and the vector b (of length n), write a code that
finds whether the system has a unique solution, infinite solutions, or no
solutions. Furthermore, the code must compute the solution if it is unique,
characterize the complete solution if there are infinitely many, and provide
the best fit solution along with the squared error if there are no solutions.
'''
import numpy as np
import numpy.linalg as la
A = np.random.rand(5, 11)
b = np.random.rand(5)



def linear_system(A, b):
    m, n = A.shape
    if b.shape[0] != m:
        print("Mismatch : Number of rows in A must match length of b.")
        return None
    
    r = la.matrix_rank(A)
    r_Ab = la.matrix_rank(np.column_stack((A, b)))

    if r == r_Ab:
        #Solutions will exist only if the augmented matrix A|b is a full rank matrix
        if r == n:
            #Unique Solutions
            print("System of Equations has a unique solution")
            x, residuals, rank, s = la.lstsq(A, b, rcond = None)
            print("Squared Error : ", np.sum(residuals) if residuals.size > 0 else 0.0)
            print(x)
        elif r != n:
            #Infinite Solutions
            print("System has infinite solutions")
            x, residuals, rank, s = la.lstsq(A, b, rcond = None)
            print("Least-squared solution : \n", x)
            print("Squared Error : ", np.sum(residuals) if residuals.size > 0 else 0.0)
            U, S, Vt = la.svd(A)
            print("Basis for null space of A : \n", Vt.T[:, np.sum(s > 1e-8):])
    else:
        #No Solutions
        print("System Has no Solutions")
        x, residuals, rank, s = la.lstsq(A, b, rcond = None)
        print("Least-Squared solution : \n", x.T)
        print("Squared Error : ", np.sum(residuals) if residuals.size > 0 else 0.0)
    return "Solved!"

'''
Note :
x, residuals, rank, s = la.lstsq(A, b, rcond = None)

gives, 
    x : closest possible solution
    residuals : b - a @ x
    rank : Rank of A
    s : Singular values of A

    From Documentation :

    Computes the vector x that approximately solves the equation a @ x = b. 
    The equation may be under-, well-, or over-determined 
    (i.e., the number of linearly independent rows of a can be less than, equal to, 
    or greater than its number of linearly independent columns). 
    If a is square and of full rank, then x (but for round-off error) is the “exact” 
    solution of the equation. Else, x minimizes the Euclidean 2-norm. 
    If there are multiple minimizing solutions, the one with the smallest 2-norm 

 is returned.
'''

def linear_lstsq(A, b):
    x, r, rank, s = la.lstsq(A, b, rcond= None)
    r = np.sum(r) if r.size > 0 else 0.0
    if(np.sum(r) == 0):
        return "Solutions Exist"
    else:
        return "No Solutions"

print(linear_system(A, b))
print(linear_lstsq(A, b))
