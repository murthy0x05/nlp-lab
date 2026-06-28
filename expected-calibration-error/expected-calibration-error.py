import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_pred, bin_edges, right=False)
    bin_indices = np.clip(bin_indices, 1, n_bins)
    
    ece = 0.0
    n = len(y_true)
    
    for b in range(1, n_bins + 1):
        mask = (bin_indices == b)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_accuracy = np.mean(y_true[mask])
            bin_confidence = np.mean(y_pred[mask])
            
            ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)
            
    return ece