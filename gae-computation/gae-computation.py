import numpy as np

def gae(rewards, values, gamma, lam):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae_cumulative = 0.0
    
    extended_values = np.append(values, 0.0)
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * extended_values[t + 1] - extended_values[t]
        
        gae_cumulative = delta + gamma * lam * gae_cumulative
        advantages[t] = gae_cumulative
        
    return advantages.tolist()