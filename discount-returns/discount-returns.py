def discount_returns(rewards: list, gamma: float) -> list:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    return returns[::-1]