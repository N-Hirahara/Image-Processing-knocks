
# coding: utf-8

# In[1]:


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


# In[2]:


# Discrete cosine transformation
T = 8
K = 4
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


# In[3]:


for yi in range(0, H, T):
    for xi in range(0, W, T):
        for v in range(T):
            for u in range(T):
                for y in range(T):
                    for x in range(T):
                        F[yi+v, xi +u] += gray_img[yi+y, xi+x] * weight(x, y, u, v)


# In[14]:


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


# In[15]:


# MSE
mse = np.mean(np.power(gray_img.astype(np.float32) - out.astype(np.float32), 2))
# PSNR
max_v = 255
psnr = 10 * np.log10(max_v**2 / mse)
print("PSNR: ", psnr)
# bitrate
bitrate =  T * (K**2) / (T**2)
print("Bitrate: ", bitrate)


# In[13]:


result(out, "37")

