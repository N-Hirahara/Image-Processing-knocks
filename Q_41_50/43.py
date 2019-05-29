
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

gray_img = 0.2126*r + 0.7152*g + 0.0772*b
gray_img.shape

# Gaussian filter
K_size = 5
s = 1.4

# 0 padding
pad = K_size // 2
gauss_out = np.zeros((pad*2+H, pad*2+W), dtype=np.float32)
gauss_out = np.pad(gray_img, (pad, pad), 'edge')
tmp = gauss_out.copy()

K_gauss = np.zeros((K_size, K_size), dtype=np.float32)
for y in range(-pad, -pad+K_size):
    for x in range(-pad, -pad+K_size):
        K_gauss[pad+y, pad+x] = np.exp( -(x**2 + y**2) / (2 * (s**2))) / (s * np.sqrt(2 * np.pi))
        
K_gauss /= K_gauss.sum()

for y in range(H):
    for x in range(W):
        gauss_out[pad+y, pad+x] = np.sum(K_gauss * tmp[y:y+K_size, x:x+K_size])

        
# Sobel filter
sobel_v = np.array([ [1., 2., 1.], [0., 0., 0.], [-1., -2., -1.] ], dtype=np.float32)
sobel_h = np.array([ [1., 0., -1.], [2., 0., -2.], [1., 0., -1.] ], dtype=np.float32)

gauss_out = gauss_out[pad-1:H+pad+1, pad-1:W+pad+1]

# 0 padding
K_size = 3
pad = K_size // 2
fy = np.zeros_like(gauss_out, dtype=np.float32)
fx = np.zeros_like(gauss_out, dtype=np.float32)

for y in range(H):
    for x in range(W):
        fy[pad+y, pad+x] = np.sum(sobel_v * gauss_out[y:y+K_size, x:x+K_size])
        fx[pad+y, pad+x] = np.sum(sobel_h * gauss_out[y:y+K_size, x:x+K_size])
        
fx = fx[pad:pad+H, pad:pad+W]
fy = fy[pad:pad+H, pad:pad+W]

edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
tan = np.arctan(fy / fx)


edge = edge.astype(np.uint8)

angle = np.zeros_like(tan, dtype=np.uint8)
angle[np.where((-0.4142 < tan) & (tan <= 0.4142))] = 0
angle[np.where((0.4142 < tan) & (tan < 2.4142))] = 45
angle[np.where((2.4142 <= tan) | (tan <= -2.4142))] = 90
angle[np.where((-2.4142 < tan) & (tan <= -0.4142))] = 135
angle = angle.astype(np.uint8)

# Non-maximum suppression による細線化
for y in range(H):
    for x in range(W):
        if angle[y, x] == 0:
            dx1, dx2, dy1, dy2 = -1, 1, 0, 0
        if angle[y, x] == 45:
            dx1, dx2, dy1, dy2 = -1, 1, 1, -1
        if angle[y, x] == 90:
            dx1, dx2, dy1, dy2 =  0, 0, -1, 1
        if angle[y, x] == 135:
            dx1, dx2, dy1, dy2 = -1, 1, -1, 1
        if x == 0:
            dx1 = 0
        if x == W-1:
            dx2 = 0
        if y == 0:
            dy1 = max(dy1, 0)
            dy2 = max(dy2, 0)
        if y == H-1:
            dy1 = min(dy1, 0)
            dy2 = min(dy2, 0)            
        if max(max(edge[y, x], edge[y+dy1,x+dx1]), edge[y+dy2, x+dx2]) != edge[y, x]:
            edge[y, x] = 0
            


# In[7]:


# ヒステリシス閾値処理
HT = 100
LT = 30

out = edge.copy()
out[out >= HT] = 255
out[out <= LT] = 0
check = np.array(np.where((LT < out) & (out < HT)))
for i in range(len(check.T)):
    y, x = check[0][i], check[1][i]
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            X = max(min(0, x+dx), W-1)
            Y = max(min(0, y+dy), H-1)
            if (out[Y, X] >= HT):
                out[y, x] = 255
            else:
                out[y, x] = 0


# In[8]:


out = out.astype(np.uint8)


# In[9]:


result(out, "43")

