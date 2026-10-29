def differencing(series: list, order: int) -> list:
    """
    Returns the series after the requested differencing order.
    """

    if order > 0:
        return differencing([series[i] - series[i - 1] for i in range(1, len(series))], order - 1)
    else:
        return series