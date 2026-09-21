def lag_features(series: list, lags: list) -> list:
    max_lag = max(lags)
    return [[series[i - l] for l in lags] for i in range(max_lag, len(series))]