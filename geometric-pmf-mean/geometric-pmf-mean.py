import numpy as np

def geometric_pmf_mean(k, p):
    P = np.power(1 - p, np.array(k, dtype=float) - 1) * p
    return [P, 1 / p]