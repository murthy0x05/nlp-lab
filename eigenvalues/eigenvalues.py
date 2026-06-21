import numpy as np

def calculate_eigenvalues(matrix):
    try:
        A = np.asarray(matrix)
    except (ValueError, TypeError):
        return None

    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.size == 0:
        return None

    try:
        eigenvalues = np.linalg.eig(A)[0]
        sort_indices = np.lexsort((np.imag(eigenvalues), np.real(eigenvalues)))
        
        return eigenvalues[sort_indices]
        
    except (np.linalg.LinAlgError, TypeError, ValueError):
        return None
