
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
gray_img = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:, 0]
gray_img = np.pad(gray_img, (1,1), 'edge')

# gradients
gx = gray_img[1:H+1, 2:] - gray_img[1:H+1, :W]
gy = gray_img[2:, 1:W+1] - gray_img[:H, 1:W+1]

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


# In[3]:


## Draw
out = gray_img[1:H+1, 1:W+1].copy().astype(np.uint8)

for y in range(cell_H):
    for x in range(cell_W):
        cx = x * N + N // 2
        cy = y * N + N // 2
        x1 = cx + N // 2 - 1
        y1 = cy
        x2 = cx - N // 2 + 1
        y2 = cy

        h = hist[y, x] / np.sum(hist[y, x])
        h /= h.max()
        
        for c in range(9):
            #angle = (20 * c + 10 - 90) / 180. * np.pi
            angle = (20 * c + 10) / 180. * np.pi
            rx = int(np.sin(angle) * (x1 - cx) + np.cos(angle) * (y1 - cy) + cx)
            ry = int(np.cos(angle) * (x1 - cx) - np.cos(angle) * (y1 - cy) + cy)
            lx = int(np.sin(angle) * (x2 - cx) + np.cos(angle) * (y2 - cy) + cx)
            ly = int(np.cos(angle) * (x2 - cx) - np.cos(angle) * (y2 - cy) + cy)
        
            c = int(255. * h[c])
            cv2.line(out, (lx, ly), (rx, ry), (c, c, c), thickness=1)


# In[4]:


result(out, "69")

