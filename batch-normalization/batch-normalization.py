import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        axis = 0
    elif x.ndim == 4:
        axis = (0, 2, 3)
        
        C = x.shape[1]
        gamma = gamma.reshape(1, C, 1, 1)
        beta = beta.reshape(1, C, 1, 1)
    else:
        raise ValueError(f"Expected 2D or 4D input, got {x.ndim}D")

    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    
    return out