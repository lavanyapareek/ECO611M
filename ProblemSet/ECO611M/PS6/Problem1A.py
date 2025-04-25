'''
Consider solving the nonlinear equations x^1/5 +y^1/5 = 2 and x^1/10 + y^1/10 = 2
simultaneously. Apply Newton's method starting from (3, 3). What
do you observe?
'''
import numpy as np
import numpy.linalg as la
def f1(x):
    return x[0]**(1/5) + x[1]**(1/5) - 2
def f2(x):
    return x[0]**(1/10) + x[1]**(1/10) - 2
def Jacob(x):
    return np.array([
        [(1/5)*x[0]**(-4/5), (1/5)*x[1]**(-4/5)],
        [(1/10)*x[0]**(-9/10), (1/5)*x[1]**(-3/5)],
    ])
def newton(f1, f2, x):
    k = 0
    while f1(x)**2 + f2(x)**2 >= 1e-12:
        print(Jacob(x))
        x = x - la.pinv(Jacob(x))@np.array([f1(x), f2(x)])
        k = k + 1
        #print(x, f1(x)**2 + f2(x)**2, la.cond(Jacob(x)), k)
    return x
y = np.array([3, 3])
x = np.array([0.5, 0.5])
print(newton(f1, f2, y))
#print(newton(f1, f2, x))