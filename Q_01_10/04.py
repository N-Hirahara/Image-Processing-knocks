
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


img = cv2.imread("imori.jpg").astype(np.float)
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# translate rgb to grayscale
def rgb2grayscale(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0772*b

gray_img = rgb2grayscale(r, g, b).astype(np.uint8)


# In[9]:


H, W, C = img.shape

# Otsu's binarization
# t: threshold, BV: between-class variance
max_t = 0
max_BV = 0

for t in range(0, 255):
    data0 = gray_img[np.where(gray_img < t)]
    w0 = len(data0) / (H * W) 
    m0 = np.mean(data0) if len(data0) > 0 else 0
    data1 = gray_img[np.where(gray_img >= t)]
    w1 = len(data1) / (H * W)
    m1 = np.mean(data1) if len(data1) > 0 else 0
    BV = w0 * w1 * (m0 - m1) * (m0 - m1)
    if BV > max_BV:
        max_BV = BV
        max_t = t


# In[10]:


threshold = max_t
print("threshold:", threshold)
out = gray_img.copy()
out[out < threshold] = 0
out[out >= threshold] = 255

cv2.imwrite("04.jpg", out)
cv2.imshow("04", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

