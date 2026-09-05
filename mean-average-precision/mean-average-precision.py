import numpy as np

def mean_average_precision(y_true_list: list, y_score_list: list, k: int | None = None) -> dict:
    ap_per_query = []
    
    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        total_relevant = np.sum(y_true)
        
        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue
            
        sorted_indices = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        if k is not None:
            y_true_sorted = y_true_sorted[:k]
                
        tps = np.cumsum(y_true_sorted)
        ranks = np.arange(1, len(y_true_sorted) + 1)
        precisions = tps / ranks
        
        ap = np.sum(precisions * y_true_sorted) / total_relevant
        ap_per_query.append(float(ap))
        
    map_value = float(np.mean(ap_per_query)) if ap_per_query else 0.0
    
    return {
        "map_value": map_value,
        "ap_per_query": ap_per_query
    }