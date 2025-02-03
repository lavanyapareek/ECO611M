import numpy as np
import numpy.linalg as la
def definiteness(X):
    if not np.allclose(X, X.T):
        print("Matrix is not symmetric")
        return None
    eigenvalues, eigenvector = la.eig(X)
    print("Eigenvalues:", eigenvalues)
    if np.all(eigenvalues > 1e-8):
        print("Matrix is Positive definite")
    elif np.all(eigenvalues >= 0):
        print("Matrix is Positive semidefinite")
        x = eigenvector[:, np.argmin(eigenvalues)]
        y = np.zeros(X.shape[0])
        while np.isclose(y.T @ X @ y, 0):
            y = np.random.rand(X.shape[0])
        print("Vector such that xTAx = 0", x)
        print("Vector such that yTAy != 0", y)
        print(y.T @ X @ y)
    elif np.all(eigenvalues < -1e-8):
        print("Matrix is Negative definite")
    elif np.all(eigenvalues <= 1e-8):
        print("Matrix is Negative semidefinite")
        x = eigenvector[:, np.argmax(eigenvalues)]
        y = np.zeros(X.shape[0])
        while np.isclose(y.T @ X @ y, 0):
            y = np.random.rand(X.shape[0])
        print("Vector such that xTAx = 0", x)
        print("Vector such that yTAy != 0", y)
        print(y.T @ X @ y)
    else:
        print("Matrix is indefinite")
    return None
A = np.array([[-2, 2],[2,-2]])
#P = A @ la.inv(A.T @ A) @ A.T

definiteness(A)