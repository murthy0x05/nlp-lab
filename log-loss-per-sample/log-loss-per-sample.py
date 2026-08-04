import math

def log_loss(y_true, y_pred, eps=1e-15):
    losses = []
    for y, p in zip(y_true, y_pred):
        p = max(min(p, 1 - eps), eps)
        losses.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return losses