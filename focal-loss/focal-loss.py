import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.array(p)
    y = np.array(y)
    
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    
    loss = -np.power((1 - p), gamma) * y * np.log(p) \
           - np.power(p, gamma) * (1 - y) * np.log(1 - p)
           
    return float(np.mean(loss))