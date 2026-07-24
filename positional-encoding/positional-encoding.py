import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    R = np.empty((0, d_model), dtype=float)

    for seq_i in range(seq_len):
        pe = np.array([], dtype=float)
        for i in range((d_model + 1) // 2):
            pe = np.append(pe, np.sin(seq_i / np.power(base, 2 * i / d_model)))
            pe = np.append(pe, np.cos(seq_i / np.power(base, 2 * i / d_model)))
        if d_model & 1:
            pe = pe[:-1]

        R = np.vstack([R, pe])
        
    return R