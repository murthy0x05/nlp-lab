def image_histogram(image):
    histogram = [0] * 256
    
    for row in image:
        for pixel in row:
            histogram[int(pixel)] += 1
            
    return histogram