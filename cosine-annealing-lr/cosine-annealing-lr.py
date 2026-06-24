import math
import numpy as np

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos((math.pi * current_step) / total_steps))