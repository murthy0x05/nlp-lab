import numpy as np

def t_test_one_sample(x, mu0):
    x = np.asarray(x)

    xb = x.mean()
    n = len(x)
    s = np.sqrt((1 / (n - 1)) * np.power((x - xb), 2).sum())

    return (xb - mu0) / (s / n ** 0.5)