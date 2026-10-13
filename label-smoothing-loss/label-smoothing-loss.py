import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    k = len(predictions)
    loss = 0.0
    for i, p in enumerate(predictions):
        y_i = (1.0 - epsilon if i == target else 0.0) + (epsilon / k)
        loss -= y_i * math.log(p)
    return loss
    