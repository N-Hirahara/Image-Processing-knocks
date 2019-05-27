
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape


# In[6]:


# Affine translation (x-sharing)
dx = 30.
dy = 0.

a = 1
b = dx / H
c = dy / W
d = 1
tx = 0
ty = 0

x, y = np.meshgrid(np.arange(W), np.arange(H))


# In[16]:


new_H = int(H+dy)
new_W = int(W+dx)

new_x = a * x + b * y + tx
new_y = c * x + d * y + ty
new_x = np.minimum(np.maximum(new_x, 0), new_W-1).astype(np.int)
new_y = np.minimum(np.maximum(new_y, 0), new_H-1).astype(np.int)

out = np.zeros((new_H, new_W, C), dtype=np.float32)
out[new_y, new_x] = img[y, x]
out = out.astype(np.uint8)


# In[17]:


result(out, "31_x")


# In[23]:


# Affine translation (x-sharing)
dx = 0.
dy = 30.

a = 1
b = dx / H
c = dy / W
d = 1
tx = 0
ty = 0

x, y = np.meshgrid(np.arange(W), np.arange(H))


# In[24]:


new_H = int(H+dy)
new_W = int(W+dx)

new_x = a * x + b * y + tx
new_y = c * x + d * y + ty
new_x = np.minimum(np.maximum(new_x, 0), new_W-1).astype(np.int)
new_y = np.minimum(np.maximum(new_y, 0), new_H-1).astype(np.int)

out = np.zeros((new_H, new_W, C), dtype=np.float32)
out[new_y, new_x] = img[y, x]
out = out.astype(np.uint8)


# In[25]:


result(out, "31_y")


# In[26]:


# Affine translation (xy-sharing)
dx = 30.
dy = 30.

a = 1
b = dx / H
c = dy / W
d = 1
tx = 0
ty = 0

x, y = np.meshgrid(np.arange(W), np.arange(H))


# In[28]:


new_H = int(H+dy)
new_W = int(W+dx)

new_x = a * x + b * y + tx
new_y = c * x + d * y + ty
new_x = np.minimum(np.maximum(new_x, 0), new_W-1).astype(np.int)
new_y = np.minimum(np.maximum(new_y, 0), new_H-1).astype(np.int)

out = np.zeros((new_H, new_W, C), dtype=np.float32)
out[new_y, new_x] = img[y, x]
out = out.astype(np.uint8)


# In[29]:


result(out, "31_xy")

