import numpy as np
def init(c, A, b, M = 1e20):
    m, n = A.shape

    I = np.eye(m)
    A = np.hstack((A, I))

    c = np.concat((c, np.full(m, M)))

    Bset = list(range(n, n + m))
    Nset = list(range(n))

    return m, n, A, c, Bset, Nset

def updateA(A, Bset, Nset, c):
    B = A[ :, Bset]
    N = A[ :, Nset]
    cb = c[Bset]
    cn = c[Nset]
    return B, N, cb, cn

def simplex(c, A_eq, b_eq, M=1e20):
    m, n, A, c, Bset, Nset =  init(c, A_eq, b_eq) # Number of constraints (m) and original variables (n)
    B, N, cb, cn = updateA(A, Bset, Nset, c)
    iter_count = 1
    while True:
        # Compute the basic solution
        B_inv = np.linalg.inv(B)  # Compute B⁻¹
        xb = B_inv @ b_eq  # Compute basic solution
        x = np.zeros(n + m)  # Full solution (including artificial variables)
        x[Bset] = xb  # Assign basic values
        
        # Compute reduced costs
        cn_bar = cn - cb @ (B_inv @ N)
        # Check optimality condition
        if (cn_bar >= 0).all():
            break  # Optimal solution reached
        
        # Choose entering variable (index in Nset)
        q = np.argmin(cn_bar)
        
        # Compute direction vector (B⁻¹ * aq)
        aq = B_inv @ N[:, q]
        
        # Ratio test for minimum leaving variable
        min_ratio, j = np.inf, -1
        for i in range(m):
            if aq[i] > 0:
                ratio = xb[i] / aq[i]
                if ratio < min_ratio:
                    min_ratio = ratio
                    j = i
        
        # If no valid pivot row found, problem is unbounded
        if j == -1:
            print("Unbounded problem.")
            return None
        
        # Swap basis and non-basis variables
        Bset[j], Nset[q] = Nset[q], Bset[j]
        
        # Update B, N, cb, and cn based on new basis
        B, N, cb, cn = updateA(A, Bset, Nset, c)
        
        iter_count += 1
    
    return x # Return only the original variables (excluding artificial variables)

# Example usage
c = np.array([2,15,5,6])  # Coefficients of the objective function
A_eq = np.array([[1,6,3,1], [2,-5,1,-3]])  # Constraint matrix
b_eq = np.array([2,3])  # Right-hand side values

from scipy.optimize import linprog
result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')
print(result.x)
print(simplex(c, A_eq, b_eq))



'''
Author's Note : Altough this is a bit different from what we have 
learnt in class but it works just fine, most of the times even better 
than the original. The only differnece is that I am taking the 
augmented c vector as [c | M, M, ....mtimes] to initialise a large 
error value for cn_bar, this supposedly gives solution everytime.
'''