
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


img = cv2.imread("imori_dark.jpg").astype(np.float)


# In[6]:


img.shape


# In[20]:


plt.figure(figsize=(8,5))
plt.hist(img.ravel(), bins=255, range=(0, 255), rwidth=0.7)
plt.title("Histogram of imori_dark.jpg", fontsize=14)
plt.savefig("20.png")
plt.show()

