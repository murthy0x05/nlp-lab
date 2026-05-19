import numpy as np

def expected_value_discrete(x, p):
    x = np.asarray(x)
    p = np.asarray(p)

    if p.sum() != 1:
        raise ValueError("invalid probabilities")
    return (x @ p).sum()