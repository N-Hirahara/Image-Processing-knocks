
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[8]:


img = cv2.imread("imori_noise.jpg")

H, W, C = img.shape

# median filter
F_size = 3

# 0-padding
pad = F_size // 2
out = np.zeros((H+pad*2, W+pad*2, C), dtype=np.float)
out[pad:pad+H, pad:pad+W, :] = img.copy()


# In[9]:


# median filter
out_tmp = out.copy()

for h in range(H):
    for w in range(W):
        for c in range(C):
            out[pad+h, pad+w, c] = np.median(out_tmp[h:h+F_size, w:w+F_size, c])

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)


# In[10]:


cv2.imwrite("10.jpg", out)
cv2.imshow("10", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

