import numpy as np

def auc(fpr, tpr):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    
    area = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0)
    
    return np.abs(area)