def percent_change(series: list) -> list:
    N = len(series)
    p = []

    for i in range(1, N):
        if series[i - 1] == 0:
            p.append(0.0)
        else:
            p.append((series[i] - series[i - 1]) / series[i - 1])

    return p