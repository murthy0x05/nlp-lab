import numpy as np

def rotate_around_z(points, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    points = np.asarray(points)
    print(points.shape)

    if points.ndim == 1:
        return points @ R.T
    else:
        return points @ R.T