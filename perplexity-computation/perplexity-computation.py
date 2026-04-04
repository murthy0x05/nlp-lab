import numpy as np

def perplexity(prob_distributions, actual_tokens):
    log_probs = []
    for i in range(len(actual_tokens)):
        log_probs.append(np.log(prob_distributions[i][actual_tokens[i]]))

    cross_entropy = -np.mean(log_probs)
    return np.exp(cross_entropy)