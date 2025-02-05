import numpy as np

def rref(X):
    X = np.array(X, dtype=float) 
    rows, cols = X.shape
    lead = 0
    for r in range(rows):
        if lead >= cols:
            break
        i = r
        #find piviot element
        while X[i, lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return X
        X[[i, r]] = X[[r, i]]  # Swap rows i and r
        if X[r, lead] != 0:
            X[r] = X[r] / X[r, lead]  # Normalize row
        for i in range(rows):
            if i != r:
                factor = X[i, lead]
                X[i] = X[i] - factor * X[r]  # Eliminate other rows
        lead += 1
    return X

print(rref([[1, 2, 3], [4, 5, 6], [7, 8, 7]]))

