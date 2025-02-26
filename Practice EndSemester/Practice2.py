#Optimisation using Newton-Rhapson

'''
A wandering alchemist has two secret ingredients,  x  and  y , that must be mixed in perfect harmony. 
The potion's stability is governed by the system:
x^{1/4} + y^{1/6} = 3
x^{1/3} + y^{1/5} = 2.5
Use Newton's method to find the correct amounts, starting from  (2,2) . 
Be wary! The wrong mix could turn thee into a toad.
'''
import numpy as np
import scipy.linalg as la

class Alchemist:
    @staticmethod
    def F(x):
        return x[0]**(1/5) + x[1]**(1/6) - 3

    @staticmethod
    def H(x):
        return x[0]**(1/3) + x[1]**(1/5) - 2.5  # (Fixed to match problem statement)

    @staticmethod
    def J(x):
        return np.array([
            [(1/5) * x[0] ** (-4/5), (1/6) * x[1] ** (-5/6)],
            [(1/3) * x[0] ** (-2/3), (1/5) * x[1] ** (-4/5)]
        ])

    @staticmethod
    def solve(x):
        while (Alchemist.F(x))**2 + (Alchemist.H(x))**2 >= 1e-12:
            x = x - la.pinv(Alchemist.J(x)) @ np.array([Alchemist.F(x), Alchemist.H(x)])
        return x

#print(Alchemist.solve(np.array([2.0, 2.0])))

'''
The Abyssal Descent 🔥
The entrance to the underworld is at the minimum of this cursed function:
f(x, y) = e^{-x^2 - y^2} + 3 sin(2x) cos(3y)
Use gradient descent with backtracking line search to find it, beginning from  (1,1) . 
How many steps dost thou require before the ground gives way?
'''

def FDescent(x):
    return np.exp( -x[0] ** 2 - x[1] ** 2 ) + 3 * np.sin( 2 * x[0] ) + np.cos( 3 * x[1] )
def GradDescent(x):
    return np.array([
        ( -2 * x[0] * np.exp( -x[0] ** 2 - x[1] ** 2 ) + 6 * np.cos( 2 * x[0] ) ),
        ( -2 * x[1] * np.exp( -x[0] ** 2 - x[1] ** 2 ) - 3 * np.sin( 3 * x[1] ) )
    ])
def btls(x):
    alpha = 1
    rho = 0.5
    c1 = 0.1
    d = -GradDescent(x)
    while  FDescent( x + alpha * d ) - FDescent( x ) > c1 * alpha * d.dot(GradDescent(x)):
        alpha = rho*alpha
    return alpha 
def solveD(x):
    k = 0
    while la.norm( GradDescent(x) ) >= 1e-6:
        d = -GradDescent(x)
        alpha = btls(x)
        x = x + alpha * d
        k += 1
    return x, k
#print( solveD( np.array( [ 1, 1 ] ) ) )

'''
The Twin-Headed Rosenbrock 👹
The two-headed demon guards the gate to wisdom! Solve for the minimum of:
f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2 + 50sin(x_1 x_2)
using conjugate gradient descent with backtracking line search. Start from  (0.5, 0.5) , 
and count thy trials!
'''
def TTHR(x):
    return 100 * ( x[1] - x[0] ** 2 ) ** 2 + ( 1 - x[0] ) ** 2 + 50 * np.sin( x[0] * x[1] )
def GTTHR(x):
    return np.array([
        200 * ( x[1] - x[0] ** 2 ) * (-2 * x[0]) + 2 * ( 1 - x[0] ) * ( -1 ) + 50 * x[1] * np.cos( x[0] * x[1] ),
        200 * ( x[1] - x[0] ** 2 ) + 50 * x[0] * np.cos( x[0] * x[1] ) 
    ])
def btlsGTTHR(x, d):
    alpha = 1
    rho = 0.8
    c1 = 0.1
    while TTHR(x + alpha * d) - TTHR( x ) > c1 * alpha * d.dot(GTTHR(x)) :
        alpha = rho * alpha
    return alpha
def solveTTHR(x):
    d = -GTTHR(x)
    k = 0
    while la.norm(GTTHR(x)) >= 1e-6:
        alpha = btlsGTTHR(x, d)
        x = x + alpha * d
        d = -GTTHR(x) if k%2 == 0 else -GTTHR(x) + ((GTTHR(x).dot(GTTHR(x)))/(d.dot(d)))*d
        k += 1
    return x, k, la.norm(GTTHR(x))
#print(solveTTHR(np.array([0.5, 0.5])))
'''
The King's Rationing Crisis 🍞🏰
The kingdom faces a famine! The king must allocate grain between three towns to maximize the happiness function:
max 3x_1 + 2x_2 + 4x_3
Subject to:
x_1 + x_2 + x_3 <= 100
2x_1 + x_2 <= 80
x_1, x_2, x_3 >= 0
Solve it using the Simplex method, and ensure the townsfolk do not revolt!
'''
def simplex(c, A, b):
    m, n = A.shape
    A, c, Bset, Nset = np.hstack((A, np.eye(m))), np.concat((c, np.full(m, 1e20))), list(range(n, n + m)), list(range(n))
    B, N, cb, cn = A[:, Bset], A[:, Nset], c[Bset], c[Nset]
    iter = 0
    while True:
        B_inv = la.inv(B)
        xb = B_inv @ b
        x = np.zeros(n + m)
        x[Bset] = xb
        cn_bar = cn - cb @ ( B_inv @ N)
        if (cn_bar >= 0).all():
            break
        q = np.argmin(cn_bar)
        aq = B_inv @ N[:, q]
        minratio, j = float('inf'), -1
        for i in range(m):
            if aq[i] > 0 :
                ratio = xb[i] / aq[i]
                if ratio < minratio: 
                    minratio = ratio
                    j = i
        if j == -1:
            return "Unbounded Solution."
        Bset[j], Nset[q] = Nset[q], Bset[j]
        B, N, cb, cn = A[:, Bset], A[:, Nset], c[Bset], c[Nset]
        iter += 1
    return x[:n]
c = np.array([-3, -2, -4, 0, 0])  # Coefficients of the objective function
A = np.array([[1, 1, 1, 1, 0], [2, 1, 0, 0, 1]])  # Constraint matrix
b = np.array([100, 80])  # Right-hand side values
from scipy.optimize import linprog
result = linprog(c, A_ub=A, b_ub=b, method='highs')
print(result.x)
print("Thou Maximiser ist : ", simplex(c, A, b), "Thou Maximum Value est grains ist : ", -c.T @ simplex(c, A, b))


