import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """

    values = np.array(values, dtype=float)
    transitions = np.array(transitions, dtype=float)
    rewards = np.array(rewards, dtype=float)

    n_states = len(values)
    n_actions = len(transitions[0])

    new_values = np.zeros(n_states)

    for s in range(n_states):
        action_values = []

        for a in range(n_actions):
            q_value = np.sum(
                transitions[s][a] * (rewards[s][a] + gamma * values)
            )

            action_values.append(q_value)

        new_values[s] = max(action_values)

    return new_values.tolist()