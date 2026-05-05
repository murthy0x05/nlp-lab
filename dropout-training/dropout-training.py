import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    if not (0 <= p < 1):
        raise ValueError("p must be in [0, 1).")
    
    # Convert input to numpy array
    x = np.array(x, dtype=float)
    
    # Random generator
    rand = rng.random if rng is not None else np.random.random
    
    # Generate random values
    r = rand(size=x.shape)
    
    # Keep mask
    keep_mask = r < (1 - p)
    
    # Scale factor
    scale = 1.0 / (1 - p)
    
    # Dropout pattern
    dropout_pattern = keep_mask.astype(float) * scale
    
    # Output
    output = x * dropout_pattern
    
    return output, dropout_pattern