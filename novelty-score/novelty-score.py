import numpy as np
import math

def novelty_score(recommendations, item_counts, n_users):
    R = np.asarray(recommendations)
    Rn = R.shape[0]

    IC = np.asarray(item_counts)
    novelty = (1 / Rn) * (-np.log2(IC / n_users)).sum()

    return novelty