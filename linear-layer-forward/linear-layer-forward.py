def linear_layer_forward(X: list, W: list, b: list) -> list:
    return (np.array(X) @ np.array(W) + np.array(b)).tolist()