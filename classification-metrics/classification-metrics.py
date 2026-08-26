import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = np.mean(y_true == y_pred)
    
    if average == 'binary':
        classes = [pos_label]
    else:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
    precisions, recalls, f1s = [], [], []
    total_tp = total_fp = total_fn = 0
    weights = []
    
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        weights.append(np.sum(y_true == c))
        
    if average == 'binary':
        precision, recall, f1_score = precisions[0], recalls[0], f1s[0]
        
    elif average == 'macro':
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1_score = np.mean(f1s)
        
    elif average == 'micro':
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    elif average == 'weighted':
        total_weight = np.sum(weights)
        if total_weight > 0:
            precision = np.average(precisions, weights=weights)
            recall = np.average(recalls, weights=weights)
            f1_score = np.average(f1s, weights=weights)
        else:
            precision, recall, f1_score = 0.0, 0.0, 0.0
            
    else:
        raise ValueError("average must be 'micro', 'macro', 'weighted', or 'binary'")
        
    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1_score), 6)
    }