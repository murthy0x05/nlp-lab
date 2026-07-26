import numpy as np

def clip_gradients(g, max_norm):
    g = np.array(g)

    g_norm = np.sqrt(np.power(g, 2).sum())

    if g_norm <= max_norm or max_norm <= 0:
        return g
    else:
        return g * (max_norm / g_norm)