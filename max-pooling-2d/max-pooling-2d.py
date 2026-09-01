def max_pooling_2d(X: list, pool_size: int) -> list:
    """
    Returns non-overlapping maximum-pooled windows.
    """
    if not X or not X[0] or pool_size <= 0:
        return []

    out_rows = len(X) // pool_size
    out_cols = len(X[0]) // pool_size
    pooled = []

    for i in range(out_rows):
        current_row = []
        for j in range(out_cols):
            r_start = i * pool_size
            c_start = j * pool_size
            
            window_max = max(
                X[r][c] 
                for r in range(r_start, r_start + pool_size) 
                for c in range(c_start, c_start + pool_size)
            )
            current_row.append(window_max)
            
        pooled.append(current_row)

    return pooled