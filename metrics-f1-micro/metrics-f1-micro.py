def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Return the micro-averaged F1 score rounded to four decimals.
    """
    if not y_true or not y_pred:
        return 0.0
        
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    micro_f1 = correct_predictions / len(y_true)
    
    return round(micro_f1, 4)