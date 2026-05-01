import numpy as np

def rotate_around_z(points, theta):
    """
    Rotate 3D point(s) around the Z-axis by angle theta (radians).

    Parameters:
        points : np.ndarray
            Shape (3,) for a single point or (N, 3) for multiple points
        theta : float
            Rotation angle in radians

    Returns:
        np.ndarray
            Rotated point(s) with same shape as input
    """
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    points = np.asarray(points)

    if points.ndim == 1:
        return R @ points
    else:
        return points @ R.T