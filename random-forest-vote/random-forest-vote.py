import numpy as np

def random_forest_vote(predictions):
    def majority(row):
        classes, counts = np.unique(row, return_counts=True)
        return classes[np.argmax(counts)].item()
        
    return [majority(row) for row in np.array(predictions).T]