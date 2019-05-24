
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from save import result

img = cv2.imread("imori.jpg").astype(np.float)


# In[6]:


out = img.copy()

S = np.size(img)
Z_max = np.max(img)
sum_h_z = 0

for i in range(0, 255):
    idx = np.where(img == i)
    sum_h_z += len(img[idx])
    Z = Z_max / S * sum_h_z
    out[idx] = Z
    
out = out.astype(np.uint8)


# In[12]:


plt.figure(figsize=(8,5))
plt.hist(out.ravel(), bins=255, range=(0, 255), rwidth=0.7)
plt.title("Translated histogram of imori.jpg", fontsize=14)
plt.savefig("23_hist.png")
plt.show()


# In[13]:


result(out, "23")

