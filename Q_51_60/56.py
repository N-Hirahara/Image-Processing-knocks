
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
template = cv2.imread("imori_part.jpg").astype(np.float32)
h, w, c = template.shape

# template matching by NCC
ncc = 0
idx_x = 0
idx_y = 0
for y in range(H-h):
    for x in range(W-w):
        ncc_tmp = np.sum(img[y:y+h, x:x+w] * template) / (np.sqrt(np.sum(np.power(img[y:y+h, x:x+w], 2)) * np.sqrt(np.sum(np.power(template, 2)))))
        if(ncc_tmp > ncc):
            ncc = ncc_tmp
            idx_x = x
            idx_y = y
                                                                  
img = cv2.rectangle(img, (idx_x, idx_y), (idx_x+w, idx_y+h), (0, 0, 255), 1)
img = img.astype(np.uint8)


# In[3]:


result(img, "56")

