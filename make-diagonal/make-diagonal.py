import numpy as np

def make_diagonal(v):
    N = len(v)

    return np.array([[v[i] if i == j else 0  for j in range(N)] for i in range(N)])