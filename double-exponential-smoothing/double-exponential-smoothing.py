def double_exponential_smoothing(series: list, alpha: float, beta: float) -> list:
    if not series:
        return []
    
    level = float(series[0])
    levels = [level]
    
    if len(series) == 1:
        return levels
    
    trend = float(series[1] - series[0])
    
    for t in range(1, len(series)):
        y_t = series[t]
        prev_level = level
        
        level = alpha * y_t + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        
        levels.append(level)
        
    return levels