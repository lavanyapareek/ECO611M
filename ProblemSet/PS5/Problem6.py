import numpy as np
import matplotlib.pyplot as plt

def F(x, n):
    if n % 2 == 0:
        if -1 <= x < 0:
            return ((x + 1) * (n + 1) - x**(n + 1) - 1) / (2 * n)
        else:
            return (x * (n + 1) - x**(n + 1)) / (2 * n) + 1/2
    else:
        if -1 <= x < 0:
            return ((x + 1) * (n + 1) + x**(n + 1) - 1) / (2 * n)
        else:
            return (x * (n + 1) - x**(n + 1)) / (2 * n) + 1/2

def f(x, n):
    return ((n + 1) / (2 * n)) * (1 - np.abs(x)**n)

def Finv(u, n):
    x = 2*u - 1
    g = F(x, n) - u
    while np.abs(g) >= 1e-6 :
        if f(x, n) != 0:
            x = x - g/f(x, n)
        else:
            break
        g = F(x, n) - u
    return x

# Parameters
n = 11

# Generate random samples
U = np.random.random(10000)  # Uniformly distributed values in [0,1]
X = np.array([Finv(u, n) for u in U])  # Transform using inverse CDF

# Generate the theoretical PDF
x_vals = np.linspace(-1, 1, 10000)
pdf_vals = f(x_vals, n)  # Theoretical probability density function

# Plot the results
plt.figure(figsize=(8, 5))
plt.hist(X, bins=np.arange(-1.1, 1.1, 0.01), density=True, alpha=0.6, label="Sampled Distribution")
plt.plot(x_vals, pdf_vals, 'r-', lw=2, label="Theoretical PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.title(f"Inverse Transform Sampling for n={n}")
plt.show()
