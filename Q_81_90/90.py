
# coding: utf-8

# In[31]:


import cv2
import numpy as np
from glob import glob
import os

def img_quantize(img):
    img = img // 64 * 64 + 32
    return img

data = glob("dataset/train_*.jpg")
data.sort()

# make database
database = np.zeros((len(data), 13), dtype=np.int)
for i, path in enumerate(data):
    img = img_quantize(cv2.imread(path))
    
    for j in range(4):
        database[i, j] = len(np.where(img[:, :, 0] == (64 * j + 32))[0])
        database[i, 4+j] = len(np.where(img[:, :, 1] == (64 * j + 32))[0])
        database[i, 8+j] = len(np.where(img[:, :, 0] == (64 * j + 32))[0])
        
# k-means
class_num = 2

# assign random class
np.random.seed(12)
for i in range(len(data)):
    if np.random.random() < 0.5:
        database[i, -1] = 0
    else:
        database[i, -1] = 1
        
while True:
    # calculate mean
    gs = np.zeros((class_num, 12), dtype=np.float32)

    for i in range(class_num):
        gs[i] = np.mean(database[np.where(database[:, -1] == i)[0], :12], axis=0)
        
    relabel_flag = False
    # re-labeling
    for i in range(len(data)):
        distances = np.sqrt(np.sum(np.square(np.abs(gs - database[i, :12])), axis=1))
        pred = np.argmin(distances)
        if database[i, -1] != pred:
            relabel_flag = True
            database[i, -1] = pred
    
    if not relabel_flag:
        break


# In[32]:


for i in range(len(data)):
    print(os.path.basename(data[i]),  "Pred :", database[i, -1])

