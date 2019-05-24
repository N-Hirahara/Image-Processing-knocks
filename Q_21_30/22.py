
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from save import result

img = cv2.imread("imori_dark.jpg").astype(np.float)


# In[9]:


# mean and std of the image
m = np.mean(img)
s = np.std(img)

# translate mean and std of the histogram
m0 = 128
s0 = 52

out = s0 / s * (img - m) + m0
out[out > 255] = 255
out[out < 0] = 0
out = out.astype(np.uint8)


# In[11]:


plt.figure(figsize=(8, 5))
plt.hist(out.ravel(), bins=255, range=(0, 255), rwidth=0.8)
plt.title("Translated histogram of imori_dark.jpg", fontsize=14)
plt.savefig("22_hist.png")
plt.show()


# In[12]:


result(out, "22")

