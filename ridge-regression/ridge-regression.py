import numpy as np

def ridge_regression(X, y, lam):
    X = np.array(X)
    y = np.array(y)

    return np.linalg.inv((X.T @ X) + (lam * np.eye(X.shape[1]))) @ X.T @ y