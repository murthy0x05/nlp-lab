def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    if not values:
        return []

    result = list(values)
    n = len(result)
    
    i = 0
    while i < n:
        if result[i] is None:
            start_none = i
            
            while i < n and result[i] is None:
                i += 1
            
            if start_none > 0 and i < n:
                left_val = result[start_none - 1]
                right_val = result[i]
                
                num_steps = i - start_none + 1
                step_size = (right_val - left_val) / num_steps
                
                for j in range(start_none, i):
                    multiplier = j - start_none + 1
                    result[j] = left_val + (step_size * multiplier)
        else:
            i += 1
            
    return result
