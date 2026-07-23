def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    if step >= total_steps:
        return float(final_lr)
        
    if step < warmup_steps:
        return float(initial_lr * (step / warmup_steps))
        
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    
    return float(initial_lr - progress * (initial_lr - final_lr))