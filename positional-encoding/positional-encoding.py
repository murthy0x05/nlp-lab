import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    encoded = np.zeros((seq_len, d_model), dtype=float)

    for i in range(seq_len):
        for j in range(d_model):
            if j & 1:
                encoded[i][j] = np.cos(i / (np.power(base, (j - 1) / d_model)))
            else:
                encoded[i][j] = np.sin(i / (np.power(base, j / d_model)))

    return encoded