import numpy as np

def sample_var_std(x):
    x = np.array(x)

    N = x.shape[0]
    var = (1 / (N - 1)) * np.square((x - x.mean())).sum()

    return [var, np.sqrt(var)]