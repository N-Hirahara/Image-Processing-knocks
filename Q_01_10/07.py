
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[10]:


import numpy as np
img = cv2.imread("imori.jpg")

H, W, C = img.shape
pool_h, pool_w = 8, 8


# In[11]:


N_h = int(H / pool_h)
N_w = int(W / pool_w)

out = img.copy()

# Average pooling
for h in range(N_h):
    for w in range(N_w):
        for c in range(C):
            out[pool_h * h : pool_h * (h+1), pool_w * w : pool_w * (w+1), c] = np.mean(out[pool_h * h : pool_h * (h+1), pool_w * w : pool_w * (w+1), c])


# In[15]:


cv2.imwrite("08.jpg", out)
cv2.imshow("08", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

