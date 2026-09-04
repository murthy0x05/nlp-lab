def user_based_cf_prediction(similarities: list, ratings: list) -> float:
    valid_pairs = [(s, r) for s, r in zip(similarities, ratings) if s > 0]
    if not valid_pairs:
        return 0.0
    
    numerator = sum(s * r for s, r in valid_pairs)
    denominator = sum(s for s, _ in valid_pairs)
    
    return numerator / denominator
    