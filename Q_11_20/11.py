
# coding: utf-8

# In[1]:


import cv2
import numpy as np

from save import result


# In[2]:


img = cv2.imread("imori.jpg")

H, W, C = img.shape


# In[4]:


# smoothing filter
F_size = 3

# zero padding
pad = F_size // 2
out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
out[pad:pad+H, pad:pad+W, :] = img.copy()


# In[5]:


# smoothing filter
out_tmp = out.copy()

for h in range(H):
    for w in range(W):
        for c in range(C):
            out[pad+h, pad+w, c] = np.mean(out[h:h+F_size, w:w+F_size, c])
            
out = out[pad:pad+H, pad:pad+W].astype(np.uint8)


# In[6]:


result(out, "11")


# In[8]:




