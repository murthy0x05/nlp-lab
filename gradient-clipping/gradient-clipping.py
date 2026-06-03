import numpy as np

def clip_gradients(g, max_norm):
    g_arr = np.array(g, dtype=float)
    
    if max_norm <= 0:
        return g_arr
    
    global_norm = np.sqrt(np.sum(np.square(g_arr)))
    
    if global_norm > max_norm:
        clip_coef = max_norm / global_norm
        g_arr = g_arr * clip_coef
        
    return g_arr