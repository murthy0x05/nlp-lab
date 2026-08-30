import numpy as np

def detect_skew(train_dist: dict, serving_dist: dict, threshold: float = 0.2, eps: float = 1e-10) -> dict:
    results = {}
    
    for feature, train_bins in train_dist.items():
        if feature not in serving_dist:
            continue
            
        p_train = np.array(train_bins, dtype=float)
        p_serve = np.array(serving_dist[feature], dtype=float)
        
        p_train = p_train / np.sum(p_train)
        p_serve = p_serve / np.sum(p_serve)
        
        p_train = np.maximum(p_train, eps)
        p_serve = np.maximum(p_serve, eps)
        
        psi = np.sum((p_serve - p_train) * np.log(p_serve / p_train))
        
        results[feature] = {
            'psi': float(psi),
            'skewed': bool(psi > threshold)
        }
        
    return results