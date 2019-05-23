
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from save import result

img = cv2.imread("imori.jpg")
H, W, C = img.shape


# In[2]:


# translate rgb to grayscale

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

def rgb2grayscale(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0772*b

gray_img = rgb2grayscale(r, g, b).astype(np.uint8)


# In[5]:


# max_min_filter
F_size = 3

# zero padding
pad = F_size // 2
out = np.zeros((H+pad*2, W+pad*2), dtype=np.float)
out[pad:pad+H, pad:pad+W] = gray_img.copy()

# max_min_filter
out_tmp = out.copy()

for h in range(H):
    for w in range(W):
        out[pad+h, pad+w] = np.max(out_tmp[h:h+F_size, w:w+F_size]) - np.min(out_tmp[h:h+F_size, w:w+F_size])
            
out = out[pad:pad+H, pad:pad+W].astype(np.uint8)


# In[7]:


result(out, "13")


# In[4]:


len(img.shape)

