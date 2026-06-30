import numpy as np

def color_to_grayscale(image):
    img_array = np.array(image)
    
    weights = np.array([0.299, 0.587, 0.114])
    grayscale_image = np.dot(img_array[...], weights)
    
    return grayscale_image.tolist()