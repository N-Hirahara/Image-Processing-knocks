
# coding: utf-8

# In[13]:


import cv2
import numpy as np
img = cv2.imread("imori.jpg")
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# translate rgb to grayscale
def rgb2grayscale(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0772*b

gray_img = rgb2grayscale(r, g, b).astype(np.uint8)
cv2.imwrite("02.jpg", gray_img)
cv2.imshow("02", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

