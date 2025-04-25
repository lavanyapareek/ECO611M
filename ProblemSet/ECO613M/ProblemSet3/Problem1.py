import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)


def genrate_MA(q, N):
    sigma = 2
    mu = 5
    eps = sigma * np.random.randn(N) + mu
    A = np.random.rand(q)
    X = np.zeros(N)

    for i in range(N):
        for j in range(q):
            if i - j >= 0:
                X[i] += A[j] * eps[i - j]

    theo_mean = mu * (1 + np.sum(A))
    theo_var = sigma ** 2 * ( 1 + np.sum(A ** 2) )
    theo_cov = []
    for j in range(1, q + 2):
        th_co = 0
        for i in range(0, q - j):
            th_co += A[i]*A[i + j]
        theo_cov.append(sigma ** 2 * th_co)
    return X, theo_mean, theo_var, np.array(theo_cov)


N = 100000
q = 5
X, theo_mean, theo_var, theo_cov  = genrate_MA(q, N)

print(theo_mean, theo_var, np.mean(theo_cov))

for _ in range(10):
    sample = np.random.choice(X, 1000)
    cov = []

    for j in range(1, 10 + 2):
        cov.append(np.cov(X[j:], X[:-j])[0, 1])
    print(np.mean(sample), np.var(sample), np.mean(np.array(cov)))

print(cov)
plt.plot(theo_cov)
plt.plot(cov)
plt.show()

            


