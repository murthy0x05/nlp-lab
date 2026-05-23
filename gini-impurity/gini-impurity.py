import numpy as np
from collections import Counter

def gini_impurity(y_left, y_right):
    N_left, N_right = len(y_left), len(y_right)
    N = N_left + N_right
    if N == 0:
        return 0.0

    freq_left = Counter(y_left)
    freq_right = Counter(y_right)

    gini_left = 1
    if N_left > 0:
        for cnt in freq_left.values():
            gini_left -= (cnt / N_left) ** 2
    gini_right = 1
    if N_right > 0:
        for cnt in freq_right.values():
            gini_right -= (cnt / N_right) ** 2

    return ((N_left / N) * gini_left) + ((N_right / N) * gini_right)