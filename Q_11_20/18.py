
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


# In[4]:


# Emboss filter
emboss_f = np.array([ [-2, -1, 0], [-1, 1, 1], [0, 1, 2] ])

out = filtering(gray_img, emboss_f, True)


# In[5]:


result(out, "18")

