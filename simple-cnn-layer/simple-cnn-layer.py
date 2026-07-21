import numpy as np

def conv2d(x, W, b):
    N, C, H, W_in = x.shape
    F, C_w, HH, WW = W.shape
    
    assert C == C_w, "Input channels must match weight channels"
    
    H_out = H - HH + 1
    W_out = W_in - WW + 1
    
    out = np.zeros((N, F, H_out, W_out))
    
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    x_slice = x[n, :, i:i+HH, j:j+WW]
                    
                    out[n, f, i, j] = np.sum(x_slice * W[f, :, :, :]) + b[f]
                    
    return out