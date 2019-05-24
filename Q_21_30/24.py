
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result


# In[2]:


img = cv2.imread("imori_gamma.jpg").astype(np.float)

# Gamma correction
c = 1
g = 2.2

out = img.copy()
out /= 255
out = (1/c * out) ** (1/g)
out *= 255
out = out.astype(np.uint8)


# In[3]:


result(out, "24")

