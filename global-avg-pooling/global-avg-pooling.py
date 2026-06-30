import numpy as np

def global_avg_pool(x):
    if x.ndim not in (3, 4):
        raise ValueError("Input must be 3D (C,H,W) or 4D (N,C,H,W)")
    return np.mean(x, axis=(-2, -1), dtype=np.float64)