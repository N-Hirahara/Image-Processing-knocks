
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


# In[2]:


# Prewitt filter
prewitt_v = np.array([ [-1, -1, -1], [0, 0, 0], [1, 1, 1] ])
prewitt_h = np.array([ [-1, 0, 1], [-1, 0, 1], [-1, 0, 1] ])

out_v = filtering(gray_img, prewitt_v, True)
out_h = filtering(gray_img, prewitt_h, True)


# In[3]:


result(out_v, "16_v")
result(out_h, "16_h")

