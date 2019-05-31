
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
template = cv2.imread("imori_part.jpg").astype(np.float32)
h, w, c = template.shape

# template matching by SAD
ssd = np.inf
idx_x = 0
idx_y = 0
for y in range(H-h):
    for x in range(W-w):
        ssd_tmp = np.sum(np.abs(img[y:y+h, x:x+w] - template))
        if(ssd_tmp < ssd):
            ssd = ssd_tmp
            idx_x = x
            idx_y = y
            
img = cv2.rectangle(img, (idx_x, idx_y), (idx_x+w, idx_y+h), (0, 0, 255), 1)
img = img.astype(np.uint8)


# In[2]:


result(img, "55")

