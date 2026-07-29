import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)
    
    is_single_point = points.ndim == 1
    if is_single_point:
        points = points.reshape(1, 3)
    
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_hom = np.hstack([points, ones])
    transformed_hom = points_hom.dot(T.T)
    
    w = transformed_hom[:, 3:]
    transformed_3d = transformed_hom[:, :3] / w
    
    if is_single_point:
        return transformed_3d.reshape(3,)
    
    return transformed_3d