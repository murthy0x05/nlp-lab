import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    num_samples, num_features = X.shape
    
    weights = np.zeros(num_features)
    bias = 0
    
    for _ in range(steps):
        model = np.dot(X, weights) + bias
        
        predictions = _sigmoid(model)
        
        dw = (1 / num_samples) * np.dot(X.T, (predictions - y))
        db = (1 / num_samples) * np.sum(predictions - y)
        
        weights -= lr * dw
        bias -= lr * db
        
    return weights, bias