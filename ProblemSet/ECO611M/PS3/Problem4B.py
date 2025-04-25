'''
(b) Let PX (X= i) = √i/sum(1,4,√j), i = 1, 2, 3, 4. Repeat what you did in
part (a) using rand command. Can you generate the random values
using randint command?
'''
import numpy as np
import matplotlib.pyplot as plt

#Using rand command 
X = np.random.rand(10000)
d = np.sum(np.sqrt([1,2,3,4]))
for i in range(10000):
    if X[i] <= 1/d:
        X[i] = 1
    elif X[i] > 1/d and X[i] <= 2**0.5/d + 1/d :
        X[i] = 2
    elif X[i] > 2**0.5/d + 1/d and X[i] <= 3**0.5/d + 2**0.5/d + 1/d :
        X[i] = 3
    else:
        X[i] = 4
Xr = X

#Using randint
X = np.random.randint(1, 10000, 10000)/10000
d = np.sum(np.sqrt([1,2,3,4]))
for i in range(10000):
    if X[i] <= 1/d:
        X[i] = 1
    elif X[i] > 1/d and X[i] <= 2**0.5/d + 1/d :
        X[i] = 2
    elif X[i] > 2**0.5/d + 1/d and X[i] <= 3**0.5/d + 2**0.5/d + 1/d :
        X[i] = 3
    else:
        X[i] = 4
Xn = X
plt.hist(Xn, bins=np.arange(1, int(d)) - 0.5, density=True, alpha=0.6, label="Xn", edgecolor="black")
plt.hist(Xr, bins=np.arange(1, int(d)) - 0.5, density=True, alpha=0.6, label="Xr", edgecolor="red")
plt.xlabel("X values")
plt.ylabel("Probability Density")
plt.legend()
plt.show()