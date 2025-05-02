import numpy as np
import matplotlib.pyplot as plt
def kNN(k, x_train, y_train, test = None):
    if test is None:
        test = x_train
    y_pred = []
    for x in test:
        distances = np.linalg.norm(x_train - x, axis = 1)
        NN = np.argsort(distances)[:k]
        NN_y = y_train[NN].astype(int)
        pred = np.sign(np.sum(NN_y))
        y_pred.append(pred)
    return np.array(y_pred)
X1 = np.random.multivariate_normal([0, 1], np.eye(2), 10)
X2 = np.random.multivariate_normal([0, 1], np.eye(2), 10)

XX1 = np.vstack([np.random.multivariate_normal(X1[i], 0.1 * np.eye(2), 10) for i in range(10)])
XX2 = np.vstack([np.random.multivariate_normal(X2[i], 0.1 * np.eye(2), 10) for i in range(10)])
X = np.vstack([XX1, XX2])

Y1 = np.ones(100)
Y2 = -np.ones(100)
Y = np.concatenate([Y1, Y2])

print('Accuracy : ', (1 - np.sum((Y != kNN(15, X, Y))) / len(Y)) * 100)

x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z_grid = kNN(15, X, Y, grid).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_grid, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.show()

# plt.scatter(X[Y == 1][ :, 0], X[ Y == 1][:, 1], color = 'purple')
# plt.scatter(X[Y == -1][ :, 0], X[ Y == -1][:, 1], color = 'red')
# plt.grid(True)
# plt.show()

import numpy as np
def nck(n,k):
    if k==0:
        return 1
    else:
        return n/k*nck(n-1,k-1)
x=np.zeros(10000)
pi,p1,p2=0.4,0.3,0.7
for i in range(10000):
    if np.random.rand()<pi:
        x[i]=np.random.binomial(20,p1)
    else:
        x[i]=np.random.binomial(20,p2)

pi_new,p1_new,p2_new=0.5,0.82,0.07
gamma=np.zeros(10000)
while((pi-pi_new)**2+(p1-p1_new)**2+(p2-p2_new)**2>1e-12):
    pi,p1,p2=pi_new,p1_new,p2_new
    for i in range(10000):
        gamma[i]=(pi*nck(20,x[i])*p1**(x[i])*(1-p1)**(20-x[i]))/(pi*nck(20,x[i])*p1**(x[i])*(1-p1)**(20-x[i])+(1-pi)*nck(20,x[i])*p2**(x[i])*(1-p2)**(20-x[i]))
    p1_new=np.sum(gamma*x)/np.sum(gamma)/20
    p2_new=np.sum((1-gamma)*x)/np.sum(1-gamma)/20
    pi_new=np.mean(gamma)
    
print(pi,p1,p2)

import numpy as np
from math import comb

n = 20
x = np.where(np.random.rand(10000) < 0.4, np.random.binomial(n, 0.3, 10000), np.random.binomial(n, 0.7, 10000))

pi, p1, p2 = 0.5, 0.82, 0.07
binom_coeff = np.array([comb(n, int(xi)) for xi in x])  # precompute once

while True:
    num = pi * binom_coeff * p1**x * (1 - p1)**(n - x)
    den = num + (1 - pi) * binom_coeff * p2**x * (1 - p2)**(n - x)
    gamma = num / den
    pi_new = np.mean(gamma)
    p1_new = np.sum(gamma * x) / (np.sum(gamma) * n)
    p2_new = np.sum((1 - gamma) * x) / (np.sum(1 - gamma) * n)
    if (pi - pi_new)**2 + (p1 - p1_new)**2 + (p2 - p2_new)**2 < 1e-12:
        break
    pi, p1, p2 = pi_new, p1_new, p2_new

print(pi, p1, p2)
