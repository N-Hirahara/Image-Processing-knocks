
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
from filtering import filtering, rgb2grayscale

img = cv2.imread("imori.jpg")

b=img[:, :, 0].copy()
g=img[:, :, 1].copy()
r=img[:, :, 2].copy()
gray_img = rgb2grayscale(r, g, b)


# In[5]:


# diferrential filter
## vertical
K_v = np.array([ [0, -1, 0], [0, 1, 0], [0, 0, 0] ])
## horizontal
K_h = np.array([ [0, 0, 0], [-1, 1, 0], [0, 0, 0] ])

out_v = filtering(gray_img, K_v, padding=True)
out_h = filtering(gray_img, K_h, padding=True)


# In[6]:


result(out_v, "14_v")
result(out_h, "14_h")

