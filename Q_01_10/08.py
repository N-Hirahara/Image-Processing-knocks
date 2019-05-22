
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img = cv2.imread("imori.jpg")

H, W, C = img.shape
pool_h, pool_w = 8, 8
N_h = int(H / pool_h)
N_w = int(W / pool_w)


# In[3]:


out = img.copy()

# Max pooling
for h in range(N_h):
    for w in range(N_w):
        for c in range(C):
            out[pool_h * h : pool_h *(h+1), pool_w * w : pool_w * (w+1), c] = np.max(out[pool_h * h : pool_h *(h+1), pool_w * w : pool_w * (w+1), c])


# In[ ]:


cv2.imwrite("08.jpg", out)
cv2.imshow("08", out)
cv2.

