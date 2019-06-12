
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale

img = cv2.imread("thorino.jpg").astype(np.float32)
H, W, C = img.shape

gray_img = 0.2126 * img[:,:, 2] + 0.7152 * img[:,:, 1] + 0.0722 * img[:,:, 0]
gray_img = gray_img.astype(np.uint8)


# In[3]:


# Sobel filter
sobel_v = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ], dtype=np.float32)
sobel_h = np.array([ [1, 0, -1], [2, 0, -2], [1, 0, -1] ], dtype=np.float32)

tmp = np.pad(gray_img, (1, 1), 'edge')

Ix = np.zeros_like(gray_img, dtype=np.float32)
Iy = np.zeros_like(gray_img, dtype=np.float32)

for y in range(H):
    for x in range(W):
        Ix[y, x] = np.mean(tmp[y:y+3, x:x+3] * sobel_h)
        Iy[y, x] = np.mean(tmp[y:y+3, x:x+3] * sobel_v)
        
tmp = np.pad(Ix, (1, 1), 'edge')

Ix2 = np.zeros_like(gray_img, dtype=np.float32)
IxIy = np.zeros_like(gray_img, dtype=np.float32)

for y in range(H):
    for x in range(W):
        Ix2[y, x] = np.mean(tmp[y:y+3, x:x+3] * sobel_h)
        IxIy[y, x] = np.mean(tmp[y:y+3, x:x+3] * sobel_v)

tmp = np.pad(Iy, (1, 1), 'edge')

Iy2 = np.zeros_like(gray_img, dtype=np.float32)

for y in range(H):
    for x in range(W):
        Iy2[y, x] = np.mean(tmp[y:y+3, x:x+3] * sobel_v)


# In[6]:


out = np.array((gray_img, gray_img, gray_img))
out = np.transpose(out, (1, 2, 0))

# Hessian
Hes = np.zeros((H,W))
for y in range(H):
    for x in range(W):
        Hes[y, x] = Ix2[y, x] * Iy2[y, x] - IxIy[y, x]**2
        
for y in range(H):
    for x in range(W):
        if Hes[y, x] == np.max(Hes[max(y-1,0): min(y+2,H-1), max(x-1,0):min(x+2,W-1)]) and Hes[y, x] > np.max(Hes)*0.1:
            out[y, x] = [0, 0, 255]

out = out.astype(np.uint8)


# In[7]:


result(out, "81")

