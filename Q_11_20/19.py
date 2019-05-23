
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale, filtering

img = cv2.imread("imori_noise.jpg")
b = img[:,:,0].copy()
g = img[:,:,1].copy()
r = img[:,:,2].copy()

gray_img = rgb2grayscale(r,g,b)


# In[16]:


H, W, C = gray_img.shape

# LoG filter
F_size = 5
s = 3

# zero padding
pad = F_size // 2

# Kernel of LoG
LoG_K = np.zeros((F_size, F_size))
for x in range(-pad, -pad+F_size):
    for y in range(-pad, -pad+F_size):
        LoG_K[y+pad, x+pad] = (x**2 + y**2 - s**2) * np.exp( (-(x**2 + y**2)) / (2*(s**2)) ) / (2 * np.pi * (s**6))
        
LoG_K /= LoG_K.sum()


# In[18]:


out = filtering(gray_img, LoG_K, True)


# In[19]:


result(out, "19")

