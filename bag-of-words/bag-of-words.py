import numpy as np
from collections import Counter

def bag_of_words_vector(tokens, vocab):
    token_counts = Counter(tokens)
    
    vector = np.array([token_counts[word] for word in vocab], dtype=int)
    
    return vector