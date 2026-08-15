import numpy as np

def log_transform(values):
    return np.log(1 + np.array(values)).tolist()