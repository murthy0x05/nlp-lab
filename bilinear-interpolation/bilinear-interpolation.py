def bilinear_resize(image: list, new_h: int, new_w: int) -> list:
    old_h = len(image)
    old_w = len(image[0]) if old_h > 0 else 0
    if old_h == 0 or old_w == 0:
        return []
    
    resized = []
    for i in range(new_h):
        row = []
        for j in range(new_w):
            y = i * (old_h - 1) / (new_h - 1) if new_h > 1 else 0.0
            x = j * (old_w - 1) / (new_w - 1) if new_w > 1 else 0.0
            
            y1 = int(y)
            y2 = min(y1 + 1, old_h - 1)
            x1 = int(x)
            x2 = min(x1 + 1, old_w - 1)
            
            dy = y - y1
            dx = x - x1
            
            top = (1 - dx) * image[y1][x1] + dx * image[y1][x2]
            bottom = (1 - dx) * image[y2][x1] + dx * image[y2][x2]
            val = (1 - dy) * top + dy * bottom
            row.append(val)
        resized.append(row)
    return resized