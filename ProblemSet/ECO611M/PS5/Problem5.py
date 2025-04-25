import numpy as np
import matplotlib.pyplot as plt

U = np.random.random(10000)

def F(x):
    return 1 - (1 + x)*np.exp(-x)
def f(x):
    return x*np.exp(-x)

def Finv(x, u):
    g = F(x) - u
    while np.abs(g) >= 1e-6 :
        if f(x) != 0:
            x = x - g/f(x)
        else:
            break
        g = F(x) - u
    return x

X = np.array([Finv(1, u) for u in U])
x_vals = np.arange(0, 50, 0.001)
pdf_vals = f(x_vals)
plt.plot(x_vals, pdf_vals)
plt.hist(X, bins = np.arange(0, 50, 0.1), density = True)
plt.show()
    