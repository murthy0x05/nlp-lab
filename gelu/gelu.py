import numpy as np
import math

def gelu(x):
    x = np.asarray(x, dtype=float)
    erf_vectorized = np.vectorize(math.erf)
    
    return 0.5 * x * (1.0 + erf_vectorized(x / math.sqrt(2.0)))