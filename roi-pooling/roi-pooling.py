import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    """

    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    H = len(feature_map)
    W = len(feature_map[0])

    pooled_outputs = []

    for roi in rois:
        x1, y1, x2, y2 = roi

        roi_w = max(x2 - x1, 1)
        roi_h = max(y2 - y1, 1)

        bin_w = roi_w / out_w
        bin_h = roi_h / out_h

        pooled = []

        for i in range(out_h):
            row = []

            for j in range(out_w):

                h_start = int(y1 + i * bin_h)
                h_end = int(y1 + (i + 1) * bin_h)

                w_start = int(x1 + j * bin_w)
                w_end = int(x1 + (j + 1) * bin_w)

                if h_end <= h_start:
                    h_end = h_start + 1

                if w_end <= w_start:
                    w_end = w_start + 1

                h_start = max(0, min(h_start, H - 1))
                h_end = max(1, min(h_end, H))

                w_start = max(0, min(w_start, W - 1))
                w_end = max(1, min(w_end, W))

                max_val = float('-inf')

                for h in range(h_start, h_end):
                    for w in range(w_start, w_end):
                        max_val = max(max_val, feature_map[h][w])

                row.append(max_val)

            pooled.append(row)

        pooled_outputs.append(pooled)

    return pooled_outputs