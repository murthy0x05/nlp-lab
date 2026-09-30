import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    arr = np.array(scores, dtype=float)
    seq_len = arr.shape[-1]
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    arr[..., mask] = mask_value
    return arr