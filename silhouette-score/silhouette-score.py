import numpy as np

def silhouette_score(X, labels):
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) <= 1 or len(unique_labels) >= n_samples:
        raise ValueError("Number of labels must be greater than 1 and less than n_samples.")
        
    sq_norms = np.sum(X**2, axis=1)
    distances_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * np.dot(X, X.T)
    distances = np.sqrt(np.maximum(distances_sq, 0))
    
    s_i = np.zeros(n_samples)
    
    for i in range(n_samples):
        label_i = labels[i]
        
        same_cluster_mask = (labels == label_i)
        same_cluster_count = np.sum(same_cluster_mask)
        
        if same_cluster_count == 1:
            s_i[i] = 0.0
            continue
            
        a_i = np.sum(distances[i, same_cluster_mask]) / (same_cluster_count - 1)
        
        b_i = np.inf
        for label_j in unique_labels:
            if label_j == label_i:
                continue
                
            other_cluster_mask = (labels == label_j)
            mean_dist_to_other = np.mean(distances[i, other_cluster_mask])
            
            if mean_dist_to_other < b_i:
                b_i = mean_dist_to_other
                
        s_i[i] = (b_i - a_i) / max(a_i, b_i)
        
    return np.mean(s_i)