import numpy as np

def min_max_scaling(data):
    R, C = len(data), len(data[0])

    X = np.array(data, dtype=float)
    for j in range(C):
        X_min, X_max = X[:, j].min(), X[:, j].max()
        if X_min == X_max:
            X[:, j] = np.zeros((1, ))
            continue
        for i in range(R):
            X[i][j] = (X[i][j] - X_min) / (X_max - X_min)

    return X.tolist()