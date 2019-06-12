
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from glob import glob
import os

def img_quantize(img):
    img = img // 64 * 64 + 32
    return img

train_data = glob("dataset/train_*.jpg")
train_data.sort()

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
    
# test data
test_data = glob("dataset/test_*.jpg")
test_data.sort()
correct = 0

# predict the class of test data by their color histgram
for test_path in test_data:
    test_img = img_quantize(cv2.imread(test_path))
    
    test_hist = np.zeros(12, dtype=np.int)
    for j in range(4):
        test_hist[j] = len(np.where(test_img[:, :, 0] == (64 * j + 32))[0])
        test_hist[4+j] = len(np.where(test_img[:, :, 1] == (64 * j + 32))[0])
        test_hist[8+j] = len(np.where(test_img[:, :, 0] == (64 * j + 32))[0])
        
    diffs = np.sum(np.abs(database[:, :12] - test_hist), axis=1)
    nearest_data_index = np.argmin(diffs)
    pred = database[nearest_data_index, -1]
    
    if "akahara" in test_path:
        ans = 0
    elif "madara" in test_path:
        ans = 1
    
    if pred == ans:
        correct += 1.
        
print("Accuracy >>", correct / len(test_data))

