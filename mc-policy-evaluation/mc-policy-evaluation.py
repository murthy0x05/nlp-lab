import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    V = np.zeros(n_states)
    sum_returns = np.zeros(n_states)
    count_returns = np.zeros(n_states)
    
    for episode in episodes:
        G = 0.0
        states_in_episode = [step[0] for step in episode]
        
        for t in reversed(range(len(episode))):
            state, reward = episode[t] 
            
            G = gamma * G + reward
            
            if state not in states_in_episode[:t]:
                sum_returns[state] += G
                count_returns[state] += 1
                V[state] = sum_returns[state] / count_returns[state]
                
    return V