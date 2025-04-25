'''
(a) Let X be a discrete random variable whose pmf is given by  P(X=i) = i
10 , i = 1, 2, 3, 4. Generate X1, X2, . . . , X1000 i.i.d. P using
randint command. Verify if the generated values are according to PX
using histogram.
'''
import matplotlib.pyplot as plt
import numpy as np
# x = np.arange(1, 2, 0.0001)
# plt.plot(x, [1/10]*10000)
# x = np.arange(2, 3, 0.0001)
# plt.plot(x, [2/10]*10000)
# x = np.arange(3, 4, 0.0001)
# plt.plot(x, [3/10]*10000)
# x = np.arange(4, 6, 0.0001)
# plt.plot(x, [4/10]*20000)
# plt.show()


X = np.random.randint(1, 11, 1000)
for i in range(1000):
    if X[i] == 1:
        X[i] = 1
    elif X[i] >= 2 and X[i] <= 3:
        X[i] = 2
    elif X[i] >= 4 and X[i] <= 6:
        X[i] = 3
    else:
        X[i] = 4
x = np.arange(1, 11)
plt.hist(X, bins=np.arange(1, 6)-0.5, density = True)
plt.show()


    