import numpy as np

def rgb2grayscale(r, g, b):
    gray_img = 0.2126*r + 0.7152*g + 0.0772*b
    return gray_img.reshape(gray_img.shape[0], gray_img.shape[1], 1)


def filtering(img, kernel, padding=True):
    H, W, C = img.shape
    
    # kernel size
    F_size = kernel.shape[0]

    # zero padding
    pad = F_size // 2
    out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy()
    
    out_tmp = out.copy()
    
    # filtering
    for h in range(H):
        for w in range(W):
            for c in range(C):
                out[pad+h, pad+w, c] = np.sum(kernel * out_tmp[h:h+F_size, w:w+F_size, c])
    
    # rescaled from 0 to 255
    out[out < 0] = 0
    out[out > 255] = 255
    
    # the size of output is the same as that of input if padding
    # the size of output is smaller than 2*pad_size on both of hight and width if not padding
    if padding:
        out = out[pad:pad+H, pad:pad+W].astype(np.uint8)
    else:
        out = out[pad*2: pad*2+(H-pad*2), pad*2: pad*2+(W-pad*2)].astype(np.uint8)
        
    return out