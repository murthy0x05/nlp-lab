import numpy as np

def chi2_independence(C):
    C = np.array(C)

    row_sums = C.sum(axis=1, keepdims=True)
    col_sums = C.sum(axis=0, keepdims=True)
    total = C.sum()

    E = (row_sums @ col_sums) / total

    chi2 = ((C - E) ** 2 / E).sum()

    return chi2, E