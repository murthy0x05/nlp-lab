import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    cdf: float = 0.0
    for i in range(k):
        cdf += (np.exp(-lam) * (lam ** i)) / math.factorial(i)

    pmf = (np.exp(-lam) * (lam ** k)) / math.factorial(k)
    return [pmf, cdf + pmf]
        