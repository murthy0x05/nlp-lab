import numpy as np

def selu(x, lam=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717):
    x_arr = np.atleast_1d(x)
    result = np.where(x_arr > 0, lam * x_arr, lam * alpha * (np.exp(x_arr) - 1))
    return np.round(result, 4).tolist()