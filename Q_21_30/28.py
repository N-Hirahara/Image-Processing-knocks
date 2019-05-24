
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape


# In[2]:


# Affine transformation
a = 1
b = 0
c = 0
d = 1
tx = 30
ty = -30

x, y = np.meshgrid(np.arange(W), np.arange(H))


# In[3]:


x_new = a * x + b * y + tx
y_new = c * x + d * y + ty
x_new = np.minimum(np.maximum(x_new, 0), W-1).astype(np.int)
y_new = np.minimum(np.maximum(y_new, 0), H-1).astype(np.int)


# In[4]:


out = np.zeros_like(img, dtype=np.float32)
out[y_new, x_new] = img[y, x]
out = out.astype(np.uint8)


# In[5]:


result(out, "28")

