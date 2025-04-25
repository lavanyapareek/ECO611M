'''
Write a code to compute the root of the following functions using bisection
method:
(a) f (x) = x^1/3-e-x2.
(b) f (x) = x4-3x3 + 4x2 + 5x-2.
(c) f (x) = 8-12x + 86x2-121x3 + 60x4-10x5
'''
import numpy as np
import matplotlib.pyplot as plt
def fa(x):
    return x**(1/3) - np.exp(-x**2)
def fb(x):
    return x**4 - 3*x**3 + 4*x**2 + 5*x - 2
def fc(x):
    return 8 - 12*x + 86*x**2 - 121*x**3 + 60*x**4 - 10*x**5

def bisection(f, a, b, tol=1e-6):
    if f(a) * f(b) >= 0:  
        print("Bisection method fails: f(a) and f(b) must have opposite signs.")
        return None
    A =[]
    B = []
    mid = (a + b) / 2
    i = 0
    while np.abs(f(mid)) >= tol:
        mid = (a + b) / 2  
        if f(a) * f(mid) < 0:  
            b = mid  
        else:
            a = mid  
        i += 1
        A.append(a)
        B.append(b)
    plt.plot(A, label="xl")
    plt.plot(B, label="xr")
    plt.legend()
    plt.show()
    return mid, f(mid), i


print(bisection(fa, 0.1, 2))
print(bisection(fb, 0.1, 2))
print(bisection(fc, 10, 0))

