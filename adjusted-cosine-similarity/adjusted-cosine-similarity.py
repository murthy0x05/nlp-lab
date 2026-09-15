import math

def adjusted_cosine_similarity(ratings_matrix: list, item_i: int, item_j: int) -> float:
    num_users = len(ratings_matrix)
    if num_users == 0:
        return 0.0
    
    user_means = []
    for u in range(num_users):
        rated_items = [ratings_matrix[u][item] for item in range(len(ratings_matrix[u])) if ratings_matrix[u][item] is not None and ratings_matrix[u][item] != 0]
        if rated_items:
            user_means.append(sum(rated_items) / len(rated_items))
        else:
            user_means.append(0.0)
            
    numerator = 0.0
    sum_sq_i = 0.0
    sum_sq_j = 0.0
    
    for u in range(num_users):
        row = ratings_matrix[u]
        if item_i < len(row) and item_j < len(row):
            val_i = row[item_i]
            val_j = row[item_j]
            if val_i is not None and val_i != 0 and val_j is not None and val_j != 0:
                mean_u = user_means[u]
                diff_i = val_i - mean_u
                diff_j = val_j - mean_u
                numerator += diff_i * diff_j
                sum_sq_i += diff_i ** 2
                sum_sq_j += diff_j ** 2
                
    denominator = math.sqrt(sum_sq_i) * math.sqrt(sum_sq_j)
    if denominator == 0:
        return 0.0
    return numerator / denominator