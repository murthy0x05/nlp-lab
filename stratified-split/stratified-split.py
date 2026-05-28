import numpy as np

def stratified_split(X, y, test_size, rng=None):
    X = np.array(X)
    y = np.array(y)
    
    classes = np.unique(y)
    
    train_idx = []
    test_idx = []
    
    for c in classes:
        idx_c = np.where(y == c)[0]
        
        if rng is not None:
            rng.shuffle(idx_c)
        else:
            np.random.shuffle(idx_c)
            
        n_c = len(idx_c)
        n_test = int(np.round(n_c * test_size))
        
        if n_test == n_c and n_c > 1:
            n_test -= 1
        elif n_c == 1:
            n_test = 0
        
        test_idx.extend(idx_c[:n_test])
        train_idx.extend(idx_c[n_test:])

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, X_test, y_train, y_test