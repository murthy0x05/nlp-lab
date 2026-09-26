import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    def f(v):
        if v is None or (isinstance(v, str) and v.strip().lower() in ("nan", "none", "")):
            return np.nan
        try:
            return float(v)
        except:
            return np.nan
    arr = np.vectorize(f)(np.array(X, dtype=object)).astype(float)
    if arr.ndim == 1:
        val = np.nanmean(arr) if strategy == "mean" else np.nanmedian(arr)
        if np.isnan(val):
            val = 0.0
        arr[np.isnan(arr)] = val
        return arr
    for j in range(arr.shape[1]):
        col = arr[:, j]
        val = np.nanmean(col) if strategy == "mean" else np.nanmedian(col)
        if np.isnan(val):
            val = 0.0
        col[np.isnan(col)] = val
        arr[:, j] = col
    return arr