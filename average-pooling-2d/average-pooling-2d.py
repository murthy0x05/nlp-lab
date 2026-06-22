import numpy as np

def average_pooling_2d(X, pool_size):
    X = np.array(X)
    
    if isinstance(pool_size, int):
        pool_h = pool_w = pool_size
    else:
        pool_h, pool_w = pool_size
        
    h, w = X.shape
    out_h = h // pool_h
    out_w = w // pool_w
    
    X_cropped = X[:out_h * pool_h, :out_w * pool_w]
    pooled = X_cropped.reshape(out_h, pool_h, out_w, pool_w).mean(axis=(1, 3))
    
    return pooled.tolist()