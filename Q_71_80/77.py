
# coding: utf-8

# In[13]:


import cv2 
import numpy as np
from save import result

# parameters
f_size = 111
s = 10
g = 1.2
l = 10
p = 0
A = 0


# In[14]:


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


# In[15]:


out = gabor_filter - np.min(gabor_filter)
out = out / np.max(out) * 255
out = out.astype(np.uint8)


# In[16]:


result(out, "77")

