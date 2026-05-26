import numpy as np

def vector_norm_3d(v):
    v = np.array(v)
    if v.ndim == 1:
        return np.sqrt(np.power(v, 2).sum())

    return np.sqrt(np.power(v, 2).sum(axis = 1 ))