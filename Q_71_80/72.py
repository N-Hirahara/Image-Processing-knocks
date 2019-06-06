
# coding: utf-8

# In[55]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape
scaled_img = img / 255


# RGB to HSV
max_v = np.max(scaled_img, axis=2)
min_v = np.min(scaled_img, axis=2)
min_bgr = np.argmin(scaled_img, axis=2)

# Hue
Hue = np.zeros_like(max_v)
Hue[np.where(max_v == min_v)] = 0
# if min_bgr = B
idx = np.where(min_bgr == 0)
Hue[idx] = 60 * (scaled_img[:, :, 1][idx] - scaled_img[:, :, 2][idx]) / (max_v[idx] - min_v[idx]) + 60
# if min_bgr = R
idx = np.where(min_bgr == 2)
Hue[idx] = 60 * (scaled_img[:, :, 0][idx] - scaled_img[:, :, 1][idx]) / (max_v[idx] - min_v[idx]) + 180
# if min_bgr = G
idx = np.where(min_bgr == 1)
Hue[idx] = 60 * (scaled_img[:, :, 2][idx] - scaled_img[:, :, 0][idx]) / (max_v[idx] - min_v[idx]) + 300

# blue color tracking
mask = np.zeros_like(Hue, dtype=np.int)
mask[np.where((180 <= Hue) & (Hue <= 260))] = 255

# Closing processing
n_iter = 5
kernel = np.array([ [0, 1, 0], [1, 0, 1], [0, 1, 0] ], dtype=np.int)
mask_ = mask.copy()

# morphology (Dilation)
for i in range(n_iter):
    tmp = np.pad(mask_, (1,1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):            
            if (np.sum(kernel * tmp[y-1:y+2, x-1:x+2]) >= 255):
                mask_[y-1, x-1] = 255
                
# morphology (Erosion)
for i in range(n_iter):
    tmp = np.pad(mask_, (1,1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):            
            if (np.sum(kernel * tmp[y-1:y+2, x-1:x+2]) < 255*4):
                mask_[y-1, x-1] = 0

# Opening processing
# morphology (Erosion)
for i in range(n_iter):
    tmp = np.pad(mask_, (1,1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):            
            if (np.sum(kernel * tmp[y-1:y+2, x-1:x+2]) < 255*4):
                mask_[y-1, x-1] = 0
                
# morphology (Dilation)
for i in range(n_iter):
    tmp = np.pad(mask_, (1,1), 'edge')
    for y in range(1, H+1):
        for x in range(1, W+1):            
            if (np.sum(kernel * tmp[y-1:y+2, x-1:x+2]) >= 255):
                mask_[y-1, x-1] = 255


# In[56]:


mask = mask_.astype(np.uint8)
result(mask, "72_mask")


# In[57]:


mask = 1 - mask_ /255
out = img * (mask[:,:,np.newaxis])
out = out.astype(np.uint8)


# In[58]:


result(out, "72")

