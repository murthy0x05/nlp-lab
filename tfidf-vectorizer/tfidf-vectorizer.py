import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    vocabulary = set()
    for doc in tokenized_docs:
        vocabulary.update(doc)
    vocabulary = sorted(list(vocabulary))
    
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    N = len(tokenized_docs)
    
    df = Counter()
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            df[word] += 1
            
    idf = {}
    for word in vocabulary:
        idf[word] = math.log(N / df[word])
            
    tfidf_matrix = np.zeros((N, len(vocabulary)))
    
    for i, doc in enumerate(tokenized_docs):
        term_counts = Counter(doc)
        total_terms = len(doc)
        
        for word, count in term_counts.items():
            if word in word_to_idx:
                tf = count / total_terms
                j = word_to_idx[word]
                tfidf_matrix[i, j] = tf * idf[word]
                
    return tfidf_matrix, vocabulary