import math
import numpy as np

def xavier_initialization(W, fan_in, fan_out):
    L = math.sqrt(6 / (fan_in + fan_out))

    W = np.asarray(W, dtype = float)
    return L * (2 * W - 1)