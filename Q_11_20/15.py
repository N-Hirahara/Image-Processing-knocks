
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale, filtering

img = cv2.imread("imori.jpg")
b = img[:,:,0].copy()
g = img[:,:,1].copy()
r = img[:,:,2].copy()

gray_img = rgb2grayscale(r,g,b)


# In[3]:


# Sobel filter
sobel_v = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ])
sobel_h = np.array([ [1, 0, -1], [2, 0, -2], [1, 0, -1] ])

out_v = filtering(gray_img, sobel_v, True)
out_h = filtering(gray_img, sobel_h, True)


# In[4]:


result(out_v, "15_v")
result(out_h, "15_h")

