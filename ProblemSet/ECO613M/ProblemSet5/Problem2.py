import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

class DecisionTreeClassifier:
    class Node:
        def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
            self.gini = gini
            self.num_samples = num_samples
            self.num_samples_per_class = num_samples_per_class
            self.predicted_class = predicted_class
            self.feature_index = None
            self.threshold = None
            self.left = None
            self.right = None

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.n_classes_ = None
        self.classes_ = None
        self.tree_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_sample(self.tree_, x) for x in X])

    def _gini_index(self, y):
        m = len(y)
        if m == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        p = counts / m
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_gini = 1.0
        best_idx, best_thr = None, None

        for idx in range(n):
            sorted_indices = np.argsort(X[:, idx])
            thresholds = X[sorted_indices, idx]
            classes = y[sorted_indices]

            num_left = np.zeros(self.n_classes_, dtype=int)
            num_right = np.array([np.sum(classes == c) for c in self.classes_])

            for i in range(1, m):
                c = classes[i-1]
                cls_idx = np.where(self.classes_ == c)[0][0]
                num_left[cls_idx] += 1
                num_right[cls_idx] -= 1

                if thresholds[i] == thresholds[i-1]:
                    continue

                left_total = i
                right_total = m - i
                gini_left = 1.0 - sum((num_left[x]/left_total)**2 for x in range(self.n_classes_)) if left_total != 0 else 0
                gini_right = 1.0 - sum((num_right[x]/right_total)**2 for x in range(self.n_classes_)) if right_total != 0 else 0
                gini = (left_total * gini_left + right_total * gini_right) / m

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i-1]) / 2

        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == c) for c in self.classes_]
        predicted_class = self.classes_[np.argmax(num_samples_per_class)]
        node = self.Node(
            gini=self._gini_index(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if node.gini == 0.0 or (self.max_depth is not None and depth >= self.max_depth):
            return node

        idx, thr = self._best_split(X, y)
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.feature_index = idx
            node.threshold = thr
            node.left = self._build_tree(X_left, y_left, depth + 1)
            node.right = self._build_tree(X_right, y_right, depth + 1)
        return node

    def _predict_sample(self, node, x):
        while node.left or node.right:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

if __name__ == "__main__":
    N = 10
    np.random.seed(42)
    X1 = np.random.multivariate_normal([1, 0, 0], np.eye(3), N)
    X2 = np.random.multivariate_normal([0, 1, 0], np.eye(3), N)
    X3 = np.random.multivariate_normal([0, 0, 1], np.eye(3), N)

    XX1 = np.vstack([np.random.multivariate_normal(X1[k], 0.1*np.eye(3), 10) for k in range(N)])
    XX2 = np.vstack([np.random.multivariate_normal(X2[k], 0.1*np.eye(3), 10) for k in range(N)])
    XX3 = np.vstack([np.random.multivariate_normal(X3[k], 0.1*np.eye(3), 10) for k in range(N)])

    X = np.vstack([XX1, XX2, XX3])
    Y = np.concatenate([np.full(10*N, 0), np.full(10*N, 1), np.full(10*N, 2)])

    clf = DecisionTreeClassifier(max_depth=100)
    clf.fit(X, Y)
    predictions = clf.predict(X)
    print("Accuracy:", np.mean(predictions == Y))

    skclf = RandomForestClassifier(max_depth = 3).fit(X, Y)
    print(skclf.score(X, Y))

    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, 0)]
    Z = clf.predict(grid).reshape(xx.shape)

    cmap = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])
    plt.scatter(X[:,0], X[:,1], c=Y, cmap=cmap, edgecolor='k', s=20)
    plt.title("Multi-class Decision Boundaries")
    plt.colorbar(ticks=[0, 1, 2])
    plt.show()
