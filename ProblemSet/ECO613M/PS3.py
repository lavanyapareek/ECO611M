# import numpy as np
# np.random.seed(42)

# def MA(q, N, mu, sig2):
#     epsilon = np.sqrt(sig2)*np.random.randn(N + q) + mu
#     A = np.random.rand(q)
#     X = np.zeros(N)
#     for i in range(N):
#         if i < q:
#             X[i] = epsilon[i]
#         else :
#             X[i] = epsilon[i + q] + np.dot(A, epsilon[i + q - 1:i + q - q -1: -1])
#     print(mu * (1 + np.sum(A)), np.mean(X))
#     print(sig2 * (1 + np.sum((A) ** 2)), np.var(X))
#     cov = []
#     for i in range(1, q):
#         cov.append(np.cov(X[i:], X[:-i])[0,1])
#         print(sig2 * np.sum(A[:q - i] * A[i :]))
#     return X
# X = MA(10, 100000, 0.5, 2)

# cov = []
# print('-'*50)
# for i in range(1, 20):
#     print(np.cov(X[i:], X[:-i])[0,1])


import numpy as np
np.random.seed(42)
def AR(N, p, mu, sig2):
    eps = np.sqrt(sig2) * np.random.randn(N + p) + mu
    A = np.array([0.01, 0.3, 0.2])
    X = np.zeros(N + p)
    for i in range(N):
        X[i] = eps[i] + np.dot(A, np.array([X[i - 1], X[i - 2], X[i - 3]]))
    print(mu / (1 - np.sum(A)))
    gammas = np.linalg.solve(
        np.array([
            [1, -A[0], -A[1], -A[2]],
            [-A[0], 1-A[1], -A[2], 0],
            [-A[1], -A[0]-A[2], 1, 0],
            [-A[2], -A[1], -A[0], 1]
        ]), np.array([sig2, 0, 0, 0])
    )
    print(np.var(X))
    for i in range(1, 4):
        print(np.cov(X[p:][i:], X[p:][:-i])[0,1])
    print(gammas)
    return np.array(X)


print(np.mean(AR(100000, 3, 5, 2)))
