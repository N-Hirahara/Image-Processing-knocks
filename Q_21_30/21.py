
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from save import result

img = cv2.imread("imori_dark.jpg").astype(np.float)


# In[10]:


vmin = np.min(img)
vmax = np.max(img)

# translate [vmin, vmax] to [0, 255]
a = 0
b = 255


# In[11]:


img[img < vmin] = a
img[vmax < img] = b
img = (b - a) / (vmax - vmin) * (img - vmin) +a
img = img.astype(np.uint8)


# In[12]:


plt.figure(figsize=(8,5))
plt.hist(img.ravel(), bins=255, range=(0, 255), rwidth=0.8)
plt.title("Histogram of transformed imori_dark.jpg", fontsize=14)
plt.savefig("21_hist.png")
plt.show()


# In[13]:


result(img, "21")

