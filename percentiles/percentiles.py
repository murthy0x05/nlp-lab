import numpy as np

def percentiles(x, q):
    x = np.sort(np.array(x))
    q = np.array(q)

    n = len(x)

    pos = (q / 100) * (n - 1)

    lower = np.floor(pos).astype(int)
    upper = np.ceil(pos).astype(int)

    weight = pos - lower

    return x[lower] * (1 - weight) + x[upper] * weight