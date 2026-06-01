import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    stride = image_size / feature_size
    scales = np.array(scales)
    aspect_ratios = np.array(aspect_ratios)

    center_x = (np.arange(feature_size) + 0.5) * stride
    center_y = (np.arange(feature_size) + 0.5) * stride
    
    # Create the grid
    cx, cy = np.meshgrid(center_x, center_y)
    cx = cx.flatten()
    cy = cy.flatten()

    # 2. Calculate widths and heights
    widths = scales[:, None] * np.sqrt(aspect_ratios[None, :])
    heights = scales[:, None] / np.sqrt(aspect_ratios[None, :])
    
    widths = widths.flatten()
    heights = heights.flatten()

    num_centers = len(cx)
    num_shapes = len(widths)

    cx_repeated = np.repeat(cx, num_shapes)
    cy_repeated = np.repeat(cy, num_shapes)
    w_tiled = np.tile(widths, num_centers)
    h_tiled = np.tile(heights, num_centers)

    xmin = cx_repeated - (w_tiled / 2)
    ymin = cy_repeated - (h_tiled / 2)
    xmax = cx_repeated + (w_tiled / 2)
    ymax = cy_repeated + (h_tiled / 2)

    anchors = np.stack((xmin, ymin, xmax, ymax), axis=1)

    return anchors.tolist()