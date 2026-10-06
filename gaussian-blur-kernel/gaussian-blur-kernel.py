import math

def gaussian_kernel(size: int, sigma: float) -> list:
    kernel = [[0.0] * size for _ in range(size)]
    center = size // 2
    total = 0.0
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            val = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i][j] = val
            total += val
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total
    return kernel