
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from save import result

imori = cv2.imread("imori.jpg").astype(np.float32)
thorino = cv2.imread("thorino.jpg").astype(np.float32)


alpha = 0.6
out = imori * alpha + thorino * (1-alpha)
out = out.astype(np.uint8)


# In[4]:


result(out, "60")

