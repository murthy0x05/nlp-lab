import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.asarray(x)
    
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(x)
    
    samples = rng.choice(x, size=(n_bootstrap, n), replace=True)
    
    boot_means = samples.mean(axis=1)
    
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    return boot_means, lower, upper