import numpy as np

def maxpool_forward(X, pool_size, stride):
    X = np.array(X)
    
    H_in, W_in = X.shape
    
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1
    
    out = np.zeros((H_out, W_out), dtype=int)
    
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_size
            w_start = w * stride
            w_end = w_start + pool_size
            
            window = X[h_start:h_end, w_start:w_end]
            out[h, w] = np.max(window)
    
    return out.tolist()