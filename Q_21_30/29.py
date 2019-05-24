
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from save import result

img = cv2.imread("imori.jpg")
H, W, C = img.shape


# In[17]:


# Affine transformation
a = 1.3
b = 0
c = 0
d = 0.8
tx = 30
ty = -30

x, y = np.meshgrid(np.arange(W), np.arange(H))


# In[18]:


W_new = np.round(a * W).astype(np.int)
H_new = np.round(d * H).astype(np.int)
out = np.zeros((H_new, W_new, C), dtype=np.float32)
x_new, y_new = np.meshgrid(np.arange(W_new), np.arange(H_new))

x = np.round((d*x_new - b*y_new) / (a*d - b*c)).astype(np.int)
y = np.round((-c*x_new + a*y_new) / (a*d - b*c)).astype(np.int)
x = np.minimum(np.maximum(x, 0), W-1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H-1).astype(np.int)


# In[19]:


out[y_new, x_new] = img[y, x]
out = out.astype(np.uint8)


# In[5]:


result(out, "29")


# In[28]:


out2 = np.zeros_like(out, dtype=np.float32)
x_2 = x_new.copy() + tx
y_2 = y_new.copy() + ty
x_2 = np.minimum(np.maximum(x_2, 0), W_new-1).astype(np.int)
y_2 = np.minimum(np.maximum(y_2, 0), H_new-1).astype(np.int)


# In[31]:


out.shape, out2.shape


# In[32]:


out2[y_2, x_2] = out[y_new, x_new]
out2 = out2.astype(np.uint8)


# In[33]:


result(out2, "29_2")

