
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[ ]:


import numpy as np
img = cv2.imread("imori.jpg")

out = img.copy()
out = out // 64 * 64 + 32

cv2.imwrite("06.jpg", out)
cv2.imshow("06", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

