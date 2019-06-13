
# coding: utf-8

# In[2]:


import cv2 
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
img = img.reshape(-1, C)


# In[3]:


# k-means
K=5
np.random.seed(0)
i = np.random.choice(np.arange(H*W), 5, replace=False)
Class = img[i].copy()


# In[19]:


label = np.zeros((H*W), dtype=np.int)
for i in range(len(img)):
    distance = np.sqrt(np.sum(np.square(np.abs(Class - img[i])), axis=1))
    label[i] = np.argmin(distance)


# In[22]:


out = label.reshape(H, W) * 50
out = out.astype(np.uint8)


# In[23]:


result(out, "91")

