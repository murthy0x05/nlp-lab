def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    arr = np.array(V, dtype=float)
    arr[s] += alpha * (r + gamma * arr[s_next] - arr[s])
    return arr
