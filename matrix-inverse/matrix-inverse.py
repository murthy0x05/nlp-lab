import numpy as np

def matrix_inverse(A):
    A = np.asarray(A)

    if not A.shape[0] == A.shape[1]:
        return None

    if np.linalg.det(A) == 0:
        return None

    return np.linalg.inv(A)