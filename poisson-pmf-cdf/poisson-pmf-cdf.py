import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    cdf = 0
    for i in range(k):
        cdf += (np.exp(-lam) * np.power(lam, i)) / math.factorial(i)

    pdf = (np.exp(-lam) * np.power(lam, k)) / math.factorial(k)
    return [pdf, pdf + cdf]