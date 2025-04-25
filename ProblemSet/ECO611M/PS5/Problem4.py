import numpy as np
import matplotlib.pyplot as plt
def fa(x):
    return x**(1/3) - np.exp(-x**2)
def fb(x):
    return x**4 - 3*x**3 + 4*x**2 + 5*x - 2
def fc(x):
    return 8 - 12*x + 86*x**2 - 121*x**3 + 60*x**4 - 10*x**5

def fag(x):
    return 1/3*x**(-2/3) - (-2*x)*np.exp(-x**2)
def fbg(x):
    return 4*x**3 - 9*x**2 + 8*x + 5
def fcg(x):
    return -12 + 172*x - 363*x**2 + 240*x**3 - 50*x**4

def nrm(f, gf, x):
    X = []
    i = 0
    while np.abs(f(x)) >= 1e-6 :
        X.append(x)
        x = x - f(x)/gf(x)
        i += 1
    plt.plot(X, label = "x")
    plt.show()
    return f(x), x, i
print(nrm(fa, fag, 0.48))
print(nrm(fb, fbg, 0.48))
print(nrm(fc, fcg, 0.48))