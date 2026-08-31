def priority_replay_sample(priorities: list, alpha: float, beta: float) -> list:
    if not priorities:
        return [[], []]
        
    scaled_priorities = [p ** alpha for p in priorities]
    total_priority = sum(scaled_priorities)
    probabilities = [p / total_priority for p in scaled_priorities]
    
    min_prob = min(probabilities)
    normalized_weights = [(min_prob / p) ** beta for p in probabilities]
    
    return [probabilities, normalized_weights]