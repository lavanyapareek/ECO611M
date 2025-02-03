import numpy as np
import matplotlib.pyplot as plt

# X = np.zeros(100000)
# for i in range(100000):
#     u = np.random.rand()
#     if u < 0.5 :
#         X[i] = -1 + (2*u)**0.5
#     else:
#         X[i] = 1 - (2*(1-u))**0.5
# plt.hist(X, bins=np.arange(-2, 2, 0.1), density=True)
# plt.show()

Xn = np.zeros(100000)
for i in range(100000):
    u = np.random.rand()
    if u <= 1/4:
        Xn[i] = 0
    elif u > 1/4 and u < 1/2:
        Xn[i] = (4*u - 1)/3
    elif u == 1/2:
        Xn[i] = 1/2
    elif u > 1/2 and u < 3/4:
        Xn[i] = 4*u/3
    else:
        Xn[i] = 1

plt.hist(Xn, bins=np.arange(-0.05, 1, 0.05), density = True)
plt.show()