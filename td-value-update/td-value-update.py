import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    V_arr = np.array(V, dtype=float)
    V_arr[s] += alpha * (r + gamma * V_arr[s_next] - V_arr[s])
    return V_arr