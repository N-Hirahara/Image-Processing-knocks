
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale, filtering

img = cv2.imread("imori.jpg")
b = img[:,:,0].copy()
g = img[:,:,1].copy()
r = img[:,:,2].copy()

gray_img = rgb2grayscale(r,g,b)


# In[4]:


# Laplacian filter
laplacian_f = np.array([ [0, 1, 0], [1, -4, 1], [0, 1, 0] ])

out = filtering(gray_img, laplacian_f, True)


# In[5]:


result(out, "17")

