
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


# Discrete Fourier Transformation
K = W
L = H
M = W
N = H

G = np.zeros((H, W), dtype=np.complex)

x, y = np.meshgrid(np.arange(W), np.arange(H))

for l in range(L):
    for k in range(K):
        G[l, k] = np.sum(gray_img * np.exp(-2j * np.pi * (x * k / M + y * l / N))) / np.sqrt(M * N)
        
power_s = (np.abs(G) / np.abs(G).max() * 255).astype(np.uint8)


# In[6]:


result(power_s, "32_ps")


# In[7]:


#  Inverse DFT
out = np.zeros((H, W), dtype=np.float32)
x, y = np.meshgrid(np.arange(W), np.arange(H))

for h in range(H):
    for w in range(W):
        out[h, w] = np.abs(np.sum( G * np.exp( 2j * np.pi * (x * w / W + y * h / H)))) / np.sqrt(H * W)
        
# out[out > 255] = 255
out = out.astype(np.uint8)


# In[8]:


result(out, "32")

