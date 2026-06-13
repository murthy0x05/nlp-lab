import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    distances = np.sqrt(
        np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
    )

    n_train = X_train.shape[0]

    neighbors = np.argsort(distances, axis=1)[:, :min(k, n_train)]

    if k > n_train:
        pad = np.full((X_test.shape[0], k - n_train), -1, dtype=int)
        neighbors = np.hstack((neighbors, pad))

    return neighbors.astype(int)