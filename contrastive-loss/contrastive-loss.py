import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)
    y: array of shape (N,) with values in {0,1}
    margin: float > 0
    reduction: "mean" or "sum"
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]

    dist = np.linalg.norm(a - b, axis=1)

    loss = y * dist**2 + (1 - y) * np.maximum(0, margin - dist)**2

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")