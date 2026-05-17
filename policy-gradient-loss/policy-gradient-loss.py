import numpy as np

def policy_gradient_loss(log_probs, rewards, gamma):
    rewards = np.array(rewards, dtype=float)
    log_probs = np.array(log_probs, dtype=float)

    T = len(rewards)

    returns = np.zeros(T)
    G = 0.0

    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    baseline = np.mean(returns)
    advantages = returns - baseline
    loss = -np.mean(log_probs * advantages)

    return float(loss)