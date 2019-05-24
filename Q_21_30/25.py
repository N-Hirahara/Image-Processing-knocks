
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float)


# In[42]:


H, W, C = img.shape

# Nearest Neighbor
a = 1.5
new_H = int(a * H)
new_W = int(a * W)


out = np.zeros((new_H, new_W, C))

for y in range(new_H):
    for x in range(new_W):
        out[y, x] = img[np.round(y / a).astype(np.int), np.round(x / a).astype(np.int)]

out = out.astype(np.uint8)


# In[43]:


result(out, "25")

