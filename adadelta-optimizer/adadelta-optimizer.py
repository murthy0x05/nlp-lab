import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    w_arr = np.array(w, dtype=float)
    grad_arr = np.array(grad, dtype=float)
    E_grad_sq_arr = np.array(E_grad_sq, dtype=float)
    E_update_sq_arr = np.array(E_update_sq, dtype=float)
    
    new_E_grad_sq = rho * E_grad_sq_arr + (1 - rho) * np.square(grad_arr)
    
    rms_update = np.sqrt(E_update_sq_arr + eps)
    rms_grad = np.sqrt(new_E_grad_sq + eps)
    delta_w = - (rms_update / rms_grad) * grad_arr
    
    new_E_update_sq = rho * E_update_sq_arr + (1 - rho) * np.square(delta_w)
    new_w = w_arr + delta_w
    
    return {
        "new_w": new_w,
        "new_E_grad_sq": new_E_grad_sq,
        "new_E_update_sq": new_E_update_sq
    }
    