import numpy as np

def robust_scaling(values):
    X = np.array(values)
    if len(X) == 1:
        return [0]

    I = np.sort(X)
    M = np.median(I)

    Q1, Q3 = None, None
    if len(I) % 2 == 0:
        Q1 = np.median(I[:(len(I) // 2)])
        Q3 = np.median(I[(len(I) // 2):])
    else:
        Q1 = np.median(I[:(len(I) // 2)])
        Q3 = np.median(I[(len(I) // 2 + 1):])

    IQR = Q3 - Q1
    if IQR == 0:
        return X - M
    return (X - M) / IQR