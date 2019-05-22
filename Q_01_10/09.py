
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img = cv2.imread("imori.jpg")

H, W, C = img.shape

# Gaussian filter
F_size = 3
s = 1.3

# 0-padding
pad = F_size // 2
out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
out[pad : pad+H, pad : pad+W, :] = img.copy().astype(np.float)


# In[13]:


# Kernel
K = np.zeros((F_size, F_size))
for x in range(-pad, -pad+F_size):
    for y in range(-pad, -pad+F_size):
        K[pad+y, pad+x] = np.exp( -(x**2 + y**2) / (2*(s**2)) ) / (s * np.sqrt(2 * np.pi))

K = K / K.sum()


# In[14]:


K


# In[15]:


out_tmp = out.copy()

for h in range(H):
    for w in range(W):
        for c in range(C):
            out[pad+h, pad+w, c] = np.sum(K * out_tmp[h:h+F_size, w:w+F_size, c])
            
out = out[pad:pad+H, pad:pad+W].astype(np.uint8)


# In[ ]:


cv2.imwrite("09.jpg", out)
cv2.imshow("09", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

