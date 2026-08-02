import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    x1, x2, label = np.array(x1), np.array(x2), np.array(label)
    
    dot_product = np.sum(x1 * x2, axis=-1)
    norm_x1 = np.linalg.norm(x1, axis=-1)
    norm_x2 = np.linalg.norm(x2, axis=-1)
    
    cos_sim = dot_product / (norm_x1 * norm_x2 + 1e-8)
    
    loss = np.where(
        label == 1,
        1.0 - cos_sim,
        np.maximum(0.0, cos_sim - margin)
    )
    
    return np.mean(loss)