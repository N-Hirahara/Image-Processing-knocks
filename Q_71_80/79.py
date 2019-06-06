
# coding: utf-8

# In[12]:


import cv2 
import numpy as np
from save import result
from filtering import filtering, rgb2grayscale

def gabor(f_size, s, g, l, p, A):
    gabor_filter = np.zeros((f_size, f_size), dtype=np.float32)

    for y in range(f_size):
        for x in range(f_size):
            dx = x - (f_size // 2)
            dy = y - (f_size // 2)
            t = A * np.pi / 180.
            x_ = np.cos(t) * dx + np.sin(t) * dy
            y_ = -np.sin(t) * dx + np.cos(t) * dy
            gabor_filter[y, x] = np.exp( -(x_**2 + g**2 * y_**2) / (2 * s**2) ) * np.cos(2 * np.pi * x_ / (l + p))

    gabor_filter /= np.sum(np.abs(gabor_filter))
    
    return gabor_filter


# In[16]:


# parameters
f_size = 11
s = 1.5
g = 1.2
l = 3
p = 0
As = [0, 45, 90, 135]


# In[17]:


filters = []
for A in As:
    filters.append(gabor(f_size, s, g, l, p, A))

    
img = cv2.imread("imori.jpg").astype(np.float32)
b = img[:,:,0].copy()
g = img[:,:,1].copy()
r = img[:,:,2].copy()
gray_img = rgb2grayscale(r, g, b)

for i, gabor_filter in enumerate(filters):
    out = filtering(gray_img,  gabor_filter, padding=True)
    out = out / np.max(out) * 255
    out = out.astype(np.uint8)
    result(out, "79_{}".format(As[i]))

