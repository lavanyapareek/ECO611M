import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as sps
np.random.seed(42)

def mixed_dist(p, N, mu, sigma):
    delta = np.random.binomial(1, p, N)
    N1 = sigma[0] ** 0.5 * np.random.randn(N) + mu[0]
    N2 = sigma[1] ** 0.5 * np.random.randn(N) + mu[1]
    dist = np.zeros(N)
    for i in range(N):
        if delta[i] == 1 :
            dist[i] = N1[i]
        else:
            dist[i] = N2[i]
    return dist

mu = np.array([-500, 1000])
sigma = np.array([1000, 0.0001])
p = 0.467589
N = 1000000
X = mixed_dist(p, N, mu, sigma)
# plt.hist(mixed_dist(p, N, mu, sigma), bins = 1000)
# plt.show()


pi_chapeau = 1/2
sigma_chapeau = np.array([X.var(), X.var()])
mu_chapeau = np.array([random.choice(X), random.choice(X)])


max_iter = 100
tol = 1e-10
curr_iter = 0


for curr_iter in range(max_iter):

    #E : 
    f1 = pi_chapeau * sps.norm.pdf(X, mu_chapeau[0], np.sqrt(sigma_chapeau[0]))
    f2 = (1 - pi_chapeau) * sps.norm.pdf(X, mu_chapeau[1], np.sqrt(sigma_chapeau[1]))
    total = f1 + f2
    Gamma1 = f1 / total
    Gamma2 = f2 / total
    N1 = np.sum(Gamma1)
    N2 = np.sum(Gamma2)

    #M :
    mu_new = np.array([
        np.sum(Gamma1 * X) / N1,
        np.sum(Gamma2 * X) / N2
    ])
    sigma_new = np.array([
        np.sum(Gamma1 * (X - mu_new[0])**2) / N1,
        np.sum(Gamma2 * (X - mu_new[1])**2) / N2
    ])

    pi_new = N1 / (N1 + N2)

    # Check for convergence
    if (np.linalg.norm(mu_new - mu_chapeau) < tol and 
        np.linalg.norm(sigma_new - sigma_chapeau) < tol):
        print(f"Converged at iteration {curr_iter}")
        break

    mu_chapeau = mu_new
    sigma_chapeau = sigma_new
    pi_chapeau = pi_new

print(mu_chapeau, sigma_chapeau, pi_chapeau)


