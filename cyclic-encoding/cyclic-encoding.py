import math
import numpy as np

def cyclic_encoding(values: list, period: float) -> list:
    theta = (2 * math.pi * np.array(values)) / period

    S = np.sin(theta)
    C = np.cos(theta)

    result = []
    for i in range(len(S)):
        result.append([S[i], C[i]])

    return result