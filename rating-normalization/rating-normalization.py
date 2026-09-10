def rating_normalization(matrix: list) -> list:
    res = []
    for row in matrix:
        valid = [x for x in row if x != 0 and x is not None]
        mean = sum(valid) / len(valid) if valid else 0
        res.append([x - mean if x != 0 and x is not None else 0 for x in row])
    return res