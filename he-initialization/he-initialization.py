def he_initialization(W, fan_in):
    W = np.array(W, dtype = float)
    fin = np.array(fan_in, dtype = float)

    L = (np.sqrt(6 / fin).sum())
    return L * (2 * W - 1)