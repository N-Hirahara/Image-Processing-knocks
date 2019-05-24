
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from save import result

img_ = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img_.shape


# In[2]:


# Affine transformation (rotate)
A = 30
rotate = - np.pi * A / 180
a = np.cos(rotate)
b = -np.sin(rotate)
c = np.sin(rotate)
d = np.cos(rotate)
tx = 0
ty = 0

img = np.zeros((H+2, W+2, C), dtype=np.float32)
img[1:H+1, 1:W+1] = img_


# In[3]:


out = np.zeros_like(img_, dtype=np.float32)
x_new, y_new = np.meshgrid(np.arange(W), np.arange(H))

x = np.round((d*x_new - b*y_new) / (a*d - b*c)).astype(np.int)
y = np.round((-c*x_new + a*y_new) / (a*d - b*c)).astype(np.int)
x = np.minimum(np.maximum(x, 0), W+1).astype(np.int)
y = np.minimum(np.maximum(y, 0), H+1).astype(np.int)


# In[4]:


out[y_new, x_new] = img[y, x]
out = out.astype(np.uint8)


# In[5]:


result(out, "30_1")


# In[11]:


center_x = W // 2
center_y = H // 2


# In[10]:


y_new - center_y


# In[12]:


x = np.round((d*(x_new-center_x) - b*(y_new-center_y)) / (a*d - b*c)).astype(np.int)
y = np.round((-c*(x_new-center_x) + a*(y_new-center_y)) / (a*d - b*c)).astype(np.int)
x = np.minimum(np.maximum(x, 0-center_x), W+1-center_x).astype(np.int)
y = np.minimum(np.maximum(y, 0-ce), H+1).astype(np.int)


# In[14]:




