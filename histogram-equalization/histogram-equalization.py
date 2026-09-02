def histogram_equalize(image: list) -> list:
    """
    Returns the histogram-equalized grayscale image.
    """
    if not image or not image[0]:
        return []

    height = len(image)
    width = len(image[0])

    hist = [0] * 256

    for row in image:
        for pixel in row:
            hist[pixel] += 1

    cdf = [0] * 256
    cdf[0] = hist[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    cdf_min = next(x for x in cdf if x > 0)
    total = height * width

    lut = [
        round((cdf[i] - cdf_min) * 255 / (total - cdf_min))
        if total != cdf_min else 0
        for i in range(256)
    ]

    return [[lut[pixel] for pixel in row] for row in image]