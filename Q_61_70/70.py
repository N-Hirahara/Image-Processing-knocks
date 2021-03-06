
# coding: utf-8

# In[4]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
scaled_img = img / 255


# RGB to HSV
max_v = np.max(scaled_img, axis=2)
min_v = np.min(scaled_img, axis=2)
min_bgr = np.argmin(scaled_img, axis=2)

# Hue
H = np.zeros_like(max_v)
H[np.where(max_v == min_v)] = 0
# if min_bgr = B
idx = np.where(min_bgr == 0)
H[idx] = 60 * (scaled_img[:, :, 1][idx] - scaled_img[:, :, 2][idx]) / (max_v[idx] - min_v[idx]) + 60
# if min_bgr = R
idx = np.where(min_bgr == 2)
H[idx] = 60 * (scaled_img[:, :, 0][idx] - scaled_img[:, :, 1][idx]) / (max_v[idx] - min_v[idx]) + 180
# if min_bgr = G
idx = np.where(min_bgr == 1)
H[idx] = 60 * (scaled_img[:, :, 2][idx] - scaled_img[:, :, 0][idx]) / (max_v[idx] - min_v[idx]) + 300


# In[5]:


out = np.zeros((img.shape[0], img.shape[1]), dtype=np.int)
out[np.where((180 <= H) & (H <= 260))] = 255
out = out.astype(np.uint8)


# In[6]:


result(out, "70")

