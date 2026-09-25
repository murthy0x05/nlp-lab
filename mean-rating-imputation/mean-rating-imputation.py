import math

def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    if not ratings_matrix or not ratings_matrix[0]:
        return [row[:] for row in ratings_matrix]

    rows = len(ratings_matrix)
    cols = len(ratings_matrix[0])
    res = [[val for val in row] for row in ratings_matrix]
    
    def is_missing(val):
        if val is None or val == 0:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        return False

    if mode == 'user':
        for i in range(rows):
            row_vals = [res[i][j] for j in range(cols) if not is_missing(res[i][j])]
            mean = sum(row_vals) / len(row_vals) if row_vals else 0
            for j in range(cols):
                if is_missing(res[i][j]):
                    res[i][j] = mean
    elif mode == 'item':
        for j in range(cols):
            col_vals = [res[i][j] for i in range(rows) if not is_missing(res[i][j])]
            mean = sum(col_vals) / len(col_vals) if col_vals else 0
            for i in range(rows):
                if is_missing(res[i][j]):
                    res[i][j] = mean
                    
    return res