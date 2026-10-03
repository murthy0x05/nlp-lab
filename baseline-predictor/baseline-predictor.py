import math

def baseline_predict(ratings_matrix: list, target_pairs: list) -> list:
    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0]) if num_users > 0 else 0
    
    all_ratings = []
    user_ratings = {u: [] for u in range(num_users)}
    item_ratings = {i: [] for i in range(num_items)}
    
    for u in range(num_users):
        for i in range(num_items):
            val = ratings_matrix[u][i]
            if val is not None and not (isinstance(val, float) and math.isnan(val)) and val != 0:
                all_ratings.append(val)
                user_ratings[u].append(val)
                item_ratings[i].append(val)
                
    mu = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0
    
    user_biases = {}
    for u in range(num_users):
        if user_ratings[u]:
            user_biases[u] = (sum(user_ratings[u]) / len(user_ratings[u])) - mu
        else:
            user_biases[u] = 0.0
            
    item_biases = {}
    for i in range(num_items):
        if item_ratings[i]:
            item_biases[i] = (sum(item_ratings[i]) / len(item_ratings[i])) - mu
        else:
            item_biases[i] = 0.0
            
    predictions = []
    for u, i in target_pairs:
        b_u = user_biases.get(u, 0.0)
        b_i = item_biases.get(i, 0.0)
        predictions.append(mu + b_u + b_i)
        
    return predictions
