
# coding: utf-8

# In[24]:


import cv2
import numpy as np
from save import result
from glob import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def img_quantize(img):
    img = img // 64 * 64 + 32
    return img


# In[31]:


train_data = glob("dataset/train_*.jpg")
train_data.sort()

plt.figure(figsize=(15,8))
# make database
database = np.zeros((len(train_data), 13), dtype=np.int)
for i, path in enumerate(train_data):
    img = img_quantize(cv2.imread(path))
    
    for j in range(4):
        database[i, j] = len(np.where(img[:, :, 0] == (64 * j + 32))[0])
        database[i, 4+j] = len(np.where(img[:, :, 1] == (64 * j + 32))[0])
        database[i, 8+j] = len(np.where(img[:, :, 0] == (64 * j + 32))[0])
        
    if "akahara" in path:
        label = 0
    elif "madara" in path:
        label = 1
        
    database[i, -1] = label
    
    hist = img.copy() // 64    
    hist[:, :, 1] += 4
    hist[:, :, 2] += 8
    plt.subplot(2, 5, i+1)
    plt.hist(hist.ravel(), bins=12, rwidth=0.9)
    plt.title(path)


# In[32]:


print(database)

