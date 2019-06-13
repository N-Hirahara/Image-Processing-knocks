
# coding: utf-8

# In[1]:


import cv2
import numpy as np

def IoU(a, b):
    W_r1, H_r1 = a[2]-a[0], a[3]-a[1]
    W_r2, H_r2 = b[2]-b[0], b[3]-b[1]
    r1 = (W_r1) * (H_r1)
    r2 = (W_r2) * (H_r2)
    
    dw = ((W_r1 + W_r2) - (abs(b[0]-a[0]) + abs(b[2]-a[2]))) / 2 
    if dw <= 0 : 
        dw = 0
    dh = ((H_r1 + H_r2) - (abs(b[1]-a[1]) + abs(b[3]-a[3]))) / 2 
    if dh <= 0 : 
        dh = 0
    rol = dw * dh
    
    return rol / (r1 + r2 - rol)


def hog(gray_img):
    H, W = gray_img.shape
    gray_img = np.pad(gray_img, (1,1), 'edge')
        
    # gradients
    gx = gray_img[1:H+1, 2:] - gray_img[1:H+1, :W]
    gy = gray_img[2:, 1:W+1] - gray_img[:H, 1:W+1]
    gx[gx == 0] = 0.000001

    # magnitude and angle of gradients
    mag = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    ang = np.arctan(gy / gx) + np.pi/2

    # mag = (mag / np.max(mag) * 255).astype(np.uint8)
    # quantize the angle of gradients
    q_ang = np.zeros_like(ang, dtype=np.int)
    d = np.pi / 9
    for i in range(9):
        q_ang[np.where((ang >= d * i) & (ang <= d * (i+1)))] = i

    # cell
    N = 8
    cell_H = H // N
    cell_W = W // N
    hist = np.zeros((cell_H, cell_W, 9), dtype=np.float32)

    for h in range(cell_H):
        for w in range(cell_W):
            for y in range(N):
                for x in range(N):
                    hist[h, w, q_ang[N*h+y, N*w+x]] += mag[N*h+y, N*w+x]

    # normalize the histogram
    C = 3
    c = C // 2
    epsilon = 1
    for h in range(cell_H):
        for w in range(cell_W):
            hist[h, w] /= np.sqrt(np.sum(np.power(hist[max(0, h-c):min(cell_H, h+C-c), max(0, w-c):min(cell_W, w+C-c)], 2)) + epsilon)

    return hist

def resize_bilinear(img, h, w):
    H_, W_ = img.shape
    
    # 拡大画像の各座標
    new_x, new_y = np.meshgrid(np.arange(w), np.arange(h))
    # 拡大画像の各座標に対応する元画像上の座標
    orig_x = new_x / (1. * w / W_)
    orig_y = new_y / (1. * h / H_)
    
    ix = np.floor(orig_x).astype(np.int)
    iy = np.floor(orig_y).astype(np.int)
    ix = np.minimum(ix, W_-2)
    iy = np.minimum(iy, H_-2)
    
    dx = orig_x - ix
    dy = orig_y - iy
    
    out = (1-dx)*(1-dy)*img[iy,ix] + dx*(1-dy)*img[iy,ix+1] + (1-dx)*dy*img[iy+1,ix] + dx*dy*img[iy+1,ix+1]
    out[out > 255] = 255
    
    return out


# In[4]:


img = cv2.imread("imori_many.jpg").astype(np.float32)
H, W, C = img.shape

gray_img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

# resize
resize_len = 32
# HOG features
H_feature = ((resize_len // 8) ** 2) * 9

# bounding boxes [h, w]
recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

# sliding window
for y in range(0, H, 4):
    for x in range(0, W, 4):
        for rec in recs:
            dh, dw = int(rec[0]//2), int(rec[1]//2)
            x1 = max(0, x-dw)
            x2 = min(W, x+dw)
            y1 = max(0, y-dh)
            y2 = min(H, y+dh)
            focus_region = gray_img[y1:y2, x1:x2]
            focus_region = resize_bilinear(focus_region, resize_len, resize_len)
            focus_hog = hog(focus_region).ravel()

