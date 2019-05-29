
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
img = cv2.imread("imori.jpg")
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# translate rgb to grayscale
def rgb2grayscale(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0772*b

H, W, C = img.shape
gray_img = rgb2grayscale(r, g, b).reshape(H, -1)

# Otsu's binarization
# t: threshold, BV: between-class variance
max_t = 0
max_BV = 0

for t in range(0, 255):
    data0 = gray_img[np.where(gray_img < t)]
    w0 = len(data0) / (H * W) 
    m0 = np.mean(data0) if len(data0) > 0 else 0
    data1 = gray_img[np.where(gray_img >= t)]
    w1 = len(data1) / (H * W)
    m1 = np.mean(data1) if len(data1) > 0 else 0
    BV = w0 * w1 * (m0 - m1) * (m0 - m1)
    if BV > max_BV:
        max_BV = BV
        max_t = t

threshold = max_t
binary_img = gray_img.copy()
binary_img[binary_img < threshold] = 0
binary_img[binary_img >= threshold] = 255


# In[2]:


# morphology (Dilation)
n_iter = 2
kernel = np.array([ [0, 1, 0], [1, 0, 1], [0, 1, 0] ])

out = binary_img.copy()

for _ in range(n_iter):
    tmp = np.pad(out, (1,1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):            
            if (np.sum(kernel * tmp[y-1:y+2, x-1:x+2]) >= 255):
                out[y-1, x-1] = 255
                
out = out.astype(np.uint8)


# In[3]:


result(out, "47")

