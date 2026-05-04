import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    if reduction == "mean":
        return sum([max(0, margin - y_true[i] * y_score[i]) for i in range(len(y_true))]) / len(y_score)
    else:
        return sum([max(0, margin - y_true[i] * y_score[i]) for i in range(len(y_true))])