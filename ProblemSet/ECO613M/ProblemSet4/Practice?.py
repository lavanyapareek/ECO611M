import numpy as np

class SimpleDecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_error = np.inf

        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            X_sorted, y_sorted = X[sorted_idx, feature], y[sorted_idx]

            for i in range(n_samples):
                threshold = X_sorted[i]
                left = y_sorted[X_sorted <= threshold]
                right = y_sorted[X_sorted > threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                left_pred = np.sign(left.sum()) or 1
                right_pred = np.sign(right.sum()) or 1

                error = (left != left_pred).sum() + (right != right_pred).sum()

                if error < best_error:
                    best_error = error
                    self.feature_index = feature
                    self.threshold = threshold
                    self.left_value = left_pred
                    self.right_value = right_pred

    def predict(self, X):
        predictions = np.where(X[:, self.feature_index] <= self.threshold, self.left_value, self.right_value)
        return predictions

np.random.seed(42)
X = np.vstack([
    np.random.multivariate_normal([1, 1], np.eye(2), 50),
    np.random.multivariate_normal([-1, -1], np.eye(2), 50),
    np.random.multivariate_normal([-1, 1], np.eye(2), 50),
    np.random.multivariate_normal([1, -1], np.eye(2), 50)
])
y = np.ones(200)
y[100:] = -1

stump = SimpleDecisionStump()
stump.fit(X, y)

preds = stump.predict(X)

accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy:.2f}")
print(f"Split on feature {stump.feature_index} at {stump.threshold:.2f}: left -> {stump.left_value}, right -> {stump.right_value}")