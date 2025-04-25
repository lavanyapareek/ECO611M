import scipy.stats as sps
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def kurtosis(data):
    data = np.array(data)
    mu = data.mean()
    sigma = data.var()
    return np.mean(((data - mu)/(sigma) ** 0.5) ** 4) - 3

data = sps.skewnorm.rvs(a = 100, size =1000)
B = 10000
bootstrap_samples = np.random.choice(data, size = (B, len(data)), replace = True)

kur = []
for i in range(B):
    kur.append(kurtosis(bootstrap_samples[i]))
kur = np.array(kur)
X = np.array(bootstrap_samples)

lower = np.percentile(kur, 2.5)
upper = np.percentile(kur, 97.5)

print(sps.bootstrap((data,), statistic=kurtosis, confidence_level=.95, method="percentile", random_state=42, n_resamples=10000).confidence_interval)
print(lower, upper)

plt.figure(figsize=(8, 5))
plt.hist(kur, bins = 100, color='orchid', edgecolor='black', alpha=0.75)
plt.axvline(lower, color='blue', linestyle='--', label=f"2.5% = {lower:.3f}")
plt.axvline(upper, color='green', linestyle='--', label=f"97.5% = {upper:.3f}")
plt.title("Bootstrap Distribution of Kurtosis")
plt.xlabel("Kurtosis")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


