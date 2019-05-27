
# coding: utf-8

# In[16]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r= img[:, :, 2].copy()

gray_img = rgb2grayscale(r, g, b).reshape(128, -1)
result(gray_img.astype(np.uint8), "38_grayscale")


# In[10]:


# Discrete cosine transformation
T = 8
K = 8
F = np.zeros_like(gray_img, dtype=np.float32)

def weight(x, y, u, v):
    if u==0:
        cu = 1 / np.sqrt(2)
    else:
        cu = 1
    if v == 0:
        cv = 1 / np.sqrt(2)
    else:
        cv = 1
    return (2 * cu * cv / T) * np.cos((2*x+1) * u * np.pi / (2*T)) * np.cos((2*y+1) * v * np.pi / (2*T))

Q = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
              (12, 12, 14, 19, 26, 58, 60, 55),
              (14, 13, 16, 24, 40, 57, 69, 56),
              (14, 17, 22, 29, 51, 87, 80, 62),
              (18, 22, 37, 56, 68, 109, 103, 77),
              (24, 35, 55, 64, 81, 104, 113, 92),
              (49, 64, 78, 87, 103, 121, 120, 101),
              (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)


# In[11]:


for yi in range(0, H, T):
    for xi in range(0, W, T):
        for v in range(T):
            for u in range(T):
                for y in range(T):
                    for x in range(T):
                        F[yi+v, xi +u] += gray_img[yi+y, xi+x] * weight(x, y, u, v)
        F[yi:yi+T, xi:xi+T] = np.round(F[yi:yi+T, xi:xi+T] / Q) * Q


# In[12]:


# IDCT
out = np.zeros((H, W), dtype=np.float32)

for yi in range(0, H, T):
    for xi in range(0, W, T):
        for y in range(T):
            for x in range(T):
                for v in range(K):
                    for u in range(K):
                        out[yi+y, xi +x] += F[yi+v, xi+u] * weight(x, y, u, v)
                        
out[out > 255] = 255
out = out.astype(np.uint8)


# In[13]:


result(out, "38")

