
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape


# In[10]:


gray_img = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:, 0]
gray_img = np.pad(gray_img, (1,1), 'edge')

# gradients
gx = gray_img[1:H+1, 2:] - gray_img[1:H+1, :W]
gy = gray_img[2:, 1:W+1] - gray_img[:H, 1:W+1]

# magnitude and angle of gradients
mag = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
ang = np.arctan(gy / gx) + np.pi/2


# In[11]:


mag = (mag / np.max(mag) * 255).astype(np.uint8)
# quantize the angle of gradients
q_ang = np.zeros_like(ang, dtype=np.int)
d = np.pi / 9
for i in range(9):
    q_ang[np.where((ang >= d * i) & (ang <= d * (i+1)))] = i


# In[4]:


result(mag, "66_mag")


# In[12]:


out = np.zeros((H, W, C), dtype=np.uint8)
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
               [127, 127, 0], [127, 0, 127], [0, 127, 127]]

for i in range(9):
    out[q_ang == i] = colors[i]


# In[13]:


result(out, "66_ang")

