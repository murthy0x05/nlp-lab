def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """

    N = len(values)

    occ = {}
    for val in values:
        occ[val] = occ.get(val, 0) + 1

    result = []
    for val in values:
        result.append(occ[val] / N)

    return result