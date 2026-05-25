import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    labels, counts = np.unique(y_train, return_counts=True)
    
    majority_label = labels[np.argmax(counts)]
    
    num_test_samples = X_test.shape[0]
    y_pred = np.full(num_test_samples, majority_label)
    
    return y_pred