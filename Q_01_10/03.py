
# coding: utf-8

# In[1]:


import cv2


# In[2]:


import numpy as np
img = cv2.imread("imori.jpg")
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# translate rgb to grayscale
def rgb2grayscale(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0772*b

gray_img = rgb2grayscale(r, g, b).astype(np.uint8)

threshold = 128
out = gray_img.copy()
out[out < threshold] = 0
out[out >= threshold] = 255

cv2.imwrite("03.jpg", out)
cv2.imshow("03", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

