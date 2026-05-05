import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    if not (0 <= p < 1):
        raise ValueError("p must be in [0, 1).")
    
    x = np.array(x, dtype=float)
    
    rand = rng.random if rng is not None else np.random.random
    
    r = rand(size=x.shape)
    
    keep_mask = r < (1 - p)
    
    scale = 1.0 / (1 - p)
    
    dropout_pattern = keep_mask.astype(float) * scale
    
    output = x * dropout_pattern
    
    return output, dropout_pattern