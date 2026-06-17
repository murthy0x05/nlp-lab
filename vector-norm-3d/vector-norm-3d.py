import numpy as np

def vector_norm_3d(v):
    v = np.asarray(v)
    if v.ndim > 1:
        return np.sqrt(np.power(v, 2).sum(axis=1))
    else:
        return np.sqrt(np.power(v, 2).sum())