import numpy as np

def gini_impurity(y_left, y_right):
    N_l, N_r = len(y_left), len(y_right)
    N = N_l + N_r

    if N == 0:
        return 0.0

    G_l = 1
    Glset = set(y_left)
    for C in Glset:
        G_l -= (y_left.count(C) / len(y_left)) ** 2
        
    G_r = 1
    Grset = set(y_right)
    for C in Grset:
        G_r -= (y_right.count(C) / len(y_right)) ** 2

    return N_l / N * G_l + N_r / N * G_r