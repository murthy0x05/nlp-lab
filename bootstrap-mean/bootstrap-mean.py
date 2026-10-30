import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    N = len(x)

    gen = np.random.default_rng(seed = seed)
    bs = gen.choice(x, size = (n_bootstrap, N), replace = True)

    means = np.mean(bs, axis = 1)

    alpha = (1 - ci) / 2
    lower = np.percentile(means, 100 * alpha)
    upper = np.quantile(means, 1 - alpha)
    
    return {
        "bootstrap_mean": np.mean(means),
        "lower": lower,
        "upper": upper
    }