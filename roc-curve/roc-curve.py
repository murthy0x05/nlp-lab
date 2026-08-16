import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score) != 0)[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[np.inf, y_score[threshold_idxs]]
    
    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps, dtype=float)
    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps, dtype=float)
    
    return fpr, tpr, thresholds