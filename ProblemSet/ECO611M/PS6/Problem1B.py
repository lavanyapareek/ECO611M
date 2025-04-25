'''
Consider solving the nonlinear equations (x^1/5 +y^1/5)^5 = 32 and (x^1/10 + y^2/5)^4 = 16
simultaneously. Apply Newton's method starting from (3, 3). What
do you observe?
'''
import numpy as np
import numpy.linalg as la
def f1(x):
    return (x[0]**(1/5) + x[1]**(1/5))**5 - 32
def f2(x):
    return (x[0]**(1/10) + x[1]**(2/5))**4 - 16
def Jacob(x):
    return np.array([
        [
            5 * (( x[0] ** (1/5) + x[1] ** (1/5) ) ** 4) * ((1/5) * x[0] ** (-4/5)),
            5 * (( x[0] ** (1/5) + x[1] ** (1/5) ) ** 4) * ((1/5) * x[1] ** (-4/5)),
        ],
        [
            4 * (( x[0] ** (1/10) + x[1] ** (2/5) ) ** 3) * ((1/10) * x[0] ** (-9/10)),
            4 * (( x[0] ** (1/10) + x[1] ** (2/5) ) ** 3) * ((2/5) * x[1] ** (-3/5)),
        ]
    ])
def newton(f1, f2, x):
    k = 0
    while f1(x)**2 + f2(x)**2 >= 1e-12:
        x = x - la.pinv(Jacob(x))@np.array([f1(x), f2(x)])
        k = k + 1
        print(x, f1(x)**2 + f2(x)**2, la.cond(Jacob(x)), k)
    return x

x = np.array([1.1, 1.1])
y = np.array([3, 3])

print(newton(f1, f2, y))
print(newton(f1, f2, x))



'''
Condition number of the function is the real cause of faster convergence of B part.
'''