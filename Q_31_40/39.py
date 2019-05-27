
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r= img[:, :, 2].copy()

def RGB2YCbCr(r, g, b):
    Y = 0.299 * r + 0.5870 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    return Y, Cb, Cr

def YCbCr2RGB(y, cb, cr):
    R = y + (cr - 128) * 1.402
    G = y - (cb - 128) * 0.3441 - (cr - 128) * 0.7139
    B = y + (cb - 128) * 1.7718
    return R, G, B


# In[2]:


y, cb, cr = RGB2YCbCr(r, g, b)

y *= 0.7
r, g, b = YCbCr2RGB(y, cb, cr)

out = np.zeros_like(img, dtype=np.float32)
out[:, :, 0] = b
out[:, :, 1] = g
out[:, :, 2] = r
out = out.astype(np.uint8)


# In[3]:


result(out, "39")

