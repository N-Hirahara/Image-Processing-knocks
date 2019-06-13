
# coding: utf-8

# In[19]:


import cv2 
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
img = img.reshape(-1, C)

# k-means
K=5
np.random.seed(0)
i = np.random.choice(np.arange(H*W), 5, replace=False)
Class = img[i].copy()
index = np.zeros((H*W), dtype=np.int)


# In[23]:


while True:
    # labeling to each pixel
    for i in range(len(img)):
        distance = np.sqrt(np.sum(np.square(np.abs(Class - img[i])), axis=1))
        index[i] = np.argmin(distance)

    # calculate mean in each label
    newClass = np.zeros_like(Class, dtype=np.float32)
    updated = False 
    for idx in range(K):
        newClass[idx] = np.mean(img[np.where(index==idx)[0]], axis=0)
    
    if (Class != newClass).any():
        Class = newClass
        updated = True
    else:
        break


# In[33]:


out = np.zeros((H,W,C), dtype=np.float32)
out = Class[index].reshape(out.shape)
out = out.astype(np.uint8)


# In[34]:


result(out, "92")


# ## k-means (k=10)

# In[36]:


img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
img = img.reshape(-1, C)

# k-means
K=10
np.random.seed(0)
i = np.random.choice(np.arange(H*W), K, replace=False)
Class = img[i].copy()
index = np.zeros((H*W), dtype=np.int)

while True:
    # labeling to each pixel
    for i in range(len(img)):
        distance = np.sqrt(np.sum(np.square(np.abs(Class - img[i])), axis=1))
        index[i] = np.argmin(distance)

    # calculate mean in each label
    newClass = np.zeros_like(Class, dtype=np.float32)
    updated = False 
    for idx in range(K):
        newClass[idx] = np.mean(img[np.where(index==idx)[0]], axis=0)
    
    if (Class != newClass).any():
        Class = newClass
        updated = True
    else:
        break
        
out = np.zeros((H,W,C), dtype=np.float32)
out = Class[index].reshape(out.shape)
out = out.astype(np.uint8)


# In[37]:


result(out, "92_K10")


# ### try another image

# In[38]:


img = cv2.imread("madara.jpg").astype(np.float32)
H, W, C = img.shape
img = img.reshape(-1, C)

# k-means
K=5
np.random.seed(0)
i = np.random.choice(np.arange(H*W), K, replace=False)
Class = img[i].copy()
index = np.zeros((H*W), dtype=np.int)

while True:
    # labeling to each pixel
    for i in range(len(img)):
        distance = np.sqrt(np.sum(np.square(np.abs(Class - img[i])), axis=1))
        index[i] = np.argmin(distance)

    # calculate mean in each label
    newClass = np.zeros_like(Class, dtype=np.float32)
    updated = False 
    for idx in range(K):
        newClass[idx] = np.mean(img[np.where(index==idx)[0]], axis=0)
    
    if (Class != newClass).any():
        Class = newClass
        updated = True
    else:
        break
        
out = np.zeros((H,W,C), dtype=np.float32)
out = Class[index].reshape(out.shape)
out = out.astype(np.uint8)


# In[39]:


result(out, "madara")

