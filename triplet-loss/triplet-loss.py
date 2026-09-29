import numpy as np

def triplet_loss(anchor: list, positive: list, negative: list, margin: float = 1.0) -> float:
    a = np.array(anchor)
    p = np.array(positive)
    n = np.array(negative)
    pos_dist = np.sum((a - p) ** 2, axis=-1)
    neg_dist = np.sum((a - n) ** 2, axis=-1)
    loss = np.maximum(0.0, pos_dist - neg_dist + margin)
    return float(np.mean(loss))