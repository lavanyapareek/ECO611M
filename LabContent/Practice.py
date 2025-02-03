# y = sqrt(x*sqrt(x*sqrt(x*sqrt(x*sqrt(x*sqrt(x*sqrt(x*sqrt(x*sqrt(x)...)))...))))
def func(n, num):
    if n == 0:
        return num
    return (num * func(n - 1, num))**0.5

# date=int(input("Enter date: "))
# month=int(input("Enter month: "))
# day=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
# month_start=[0,3,4,0,2,5,0,3,6,1,4,6]
# print(day[(date+month_start[month-1]-1)%7])



import numpy as np
import matplotlib.pyplot as plt

# p=np.random.rand(20)
# q=p/np.sum(p)
# print("Generated pmf:",q)
# mean=np.sum(q*np.arange(1,21))
# print("Mean of the random variable:",mean)
# print("Variance:",np.sum(q*np.arange(1,21)**2)-mean**2)

# x=np.random.rand(1000)
# cdf = np.cumsum(q)
# y=np.zeros((1000))
# for i in range(1000):
# 	for j in range(20):
# 		if x[i]>=cdf[j] and x[i]<cdf[j+1]:
# 			y[i]=j+1 # y can also be computed using "np.random.choice"
# 			break
# plt.hist(y,20,density=True)
# plt.show()
# print("Sample mean:",np.mean(y))
# print("Sample variance:",np.var(y))


import numpy as np
import matplotlib.pyplot as plt

'''
X = np.zeros(10000)
d = np.sum(np.arange(1, 5)**2)
print(d)
for i in range(10000):
    u = np.random.rand()
    if u <= 1/d:
        X[i] = 1
    elif u <= 1/d + 4/d and u > 1/d:
        X[i] = 2
    elif u <= 1/d + 4/d + 9/d and u > 1/d + 4/d:
        X[i] = 3
    else:
        X[i] = 4
print(np.mean(X))
print(np.var(X))
plt.hist(X, bins = np.arange(1, 6, 1), density = True)
plt.show()
'''

# values = np.array([1,2,3,4,5,6])
# pmf = 2**values/np.sum(2**values)
# cdf = np.cumsum(pmf)
# print(pmf)
# print(cdf)
# U = np.random.rand(10000)
# for i in range(len(U)):
#     for j in range(len(cdf)):
#         if U[i] <= cdf[j]:
#             U[i] = values[j]
#             break

# plt.hist(U, bins = np.arange(1, 8, 0.6), density = True)
# plt.show()

