import numpy as np

def majority_classifier(y_train, X_test):
    majority_label = np.bincount(y_train).argmax()

    return np.full(len(X_test), majority_label)