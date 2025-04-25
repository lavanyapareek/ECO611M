import numpy as np
np.random.seed(42)
def generate_stable_ar_coeffs(p):
    def is_stationary(coefs):
        companion = np.zeros((p, p))
        companion[0, :] = coefs
        if p > 1:
            companion[1:, :-1] = np.eye(p - 1)
        return np.all(np.abs(np.linalg.eigvals(companion)) < 1)
    
    while True:
        coefs = np.random.uniform(-1, 1, p) * 0.5
        if is_stationary(coefs):
            break
    return coefs

def AR_gen(p, N, burn_in = 1000):
    N = N + burn_in
    A = generate_stable_ar_coeffs(p)
    X = np.zeros(N)
    mu = 0.5
    sigma = 2
    epsilon = sigma * np.random.randn(N) + mu
    for i in range(p, N):
        X[i] = epsilon[i]
        for j in range(p):
            X[i] += A[j] * X[i - j - 1]
    theo_mean = mu / (1 - np.sum(A))
    return X[burn_in :], theo_mean, A

def yule_walker(a, sigma):
    """
    Yule Walker :
    gamma(m) = sum(1, p){a_j * gamma(m - j)} + sigma^2(if m == 0)
    g(0) - a1 g(1) - a2 g(2) - a3 g(3) = sigma^2 
    g(1) - a1 g(0) - a2 g(1) - a3 g(2) = 0
    g(2) - a1 g(1) - a2 g(0) - a3 g(1) = 0
    g(3) - a1 g(2) - a2 g(1) - a3 g(0) = 0
    """
    A = np.array([
        # g(0)       g(1)     g(2)   g(3)
        [  1,          -a[0], -a[1], -a[2]],
        [-a[0],     1 - a[1], -a[2],   0  ],
        [-a[1], -a[0] - a[2], 1,   0  ],
        [-a[2], -a[1], -a[0], 1]
    ])
    gammas = np.linalg.solve(A, np.array([sigma**2, 0, 0, 0]))
    gammas = np.append(gammas, a[0] * gammas[3] + a[1] * gammas[2] + a[2] * gammas[1])
    gammas = np.append(gammas, a[0] * gammas[4] + a[1] * gammas[3] + a[2] * gammas[2])
    return gammas
def get_covariance(gamas, p):
    SIGMA = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            delta = np.abs(i - j)
            if delta < len(gamas):
                SIGMA[i, j] = gamas[delta]
            else:
                SIGMA[i, j] = 0
    return SIGMA
def autocovariances(x, max_lag):
    n = len(x)
    mean = np.mean(x)
    result = []
    for lag in range(max_lag + 1):
        cov = np.sum((x[:n - lag] - mean) * (x[lag:] - mean)) / (n - lag)
        result.append(cov)
    return np.array(result)

X, th_mean, A = AR_gen(3, 1000000)
print(th_mean)
theoretical_cov = get_covariance(yule_walker(A, 2), 3)
empirical_cov = autocovariances(X, 3)
print("Theoretical Cov:", theoretical_cov[0])
print("Empirical Cov:", empirical_cov)
mean = 0
for _ in range(10):
    sample = np.random.choice(X, 10000)
    print(np.mean(sample), np.var(sample))
    mean += np.mean(sample)
print(mean/100000)