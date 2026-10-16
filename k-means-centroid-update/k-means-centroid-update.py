def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    if not points:
        return []
    
    dimension = len(points[0])
    cluster_sums = [[0.0] * dimension for _ in range(k)]
    cluster_counts = [0] * k
    
    for point, cluster_id in zip(points, assignments):
        cluster_counts[cluster_id] += 1
        for i in range(dimension):
            cluster_sums[cluster_id][i] += point[i]
            
    centroids = []
    for j in range(k):
        if cluster_counts[j] > 0:
            centroid = [cluster_sums[j][i] / cluster_counts[j] for i in range(dimension)]
        else:
            centroid = [0.0] * dimension
        centroids.append(centroid)
        
    return centroids