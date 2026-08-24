import numpy as np

def pca_projection(X: list, k: int) -> list:
    X_np = np.array(X)
    
    mean = np.mean(X_np, axis=0)
    X_centered = X_np - mean
    
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    top_k_components = sorted_eigenvectors[:, :k]
    projected_data = np.dot(X_centered, top_k_components)
    
    return projected_data.tolist()