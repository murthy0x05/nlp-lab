import numpy as np

def rnn_step_backward(dh, cache):
    dh = np.array(dh)
    x_t, h_prev, h_t, W, U, b = [np.array(c) for c in cache]
    
    da = dh * (1 - h_t ** 2) 
    dx_t = np.dot(W.T, da)
    dh_prev = np.dot(U.T, da)
    dW = np.outer(da, x_t)
    dU = np.outer(da, h_prev)
    db = da 
    
    return dx_t.tolist(), dh_prev.tolist(), dW.tolist(), dU.tolist(), db.tolist()