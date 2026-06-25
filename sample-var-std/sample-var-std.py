import numpy as np

def sample_var_std(x):
    x = np.asarray(x)
    var = np.sum(np.power(x - np.mean(x), 2)) / (len(x) - 1)
    std = np.sqrt(var)

    return (var, std)