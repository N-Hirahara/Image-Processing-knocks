
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


# In[5]:


# Discrete Fourier Transformation
K = W
L = H

G = np.zeros((H, W), dtype=np.complex)

x, y = np.meshgrid(np.arange(W), np.arange(H))

for l in range(L):
    for k in range(K):
        G[l, k] = np.sum(gray_img * np.exp(-2j * np.pi * (x * k / W + y * l / H))) / np.sqrt(W * H)


# In[32]:


trans_G = np.zeros_like(G, dtype=np.complex)
trans_G[:H//2, :W//2] = G[H//2:, W//2:]
trans_G[:H//2, W//2:] = G[H//2:, :W//2]
trans_G[H//2:, :W//2] = G[:H//2, W//2:]
trans_G[H//2:, W//2:] = G[:H//2, :W//2]

# low-pass filter
p = 0.5
centerized_x = x - W // 2
centerized_y = y - H // 2
r = np.sqrt(centerized_x ** 2 + centerized_y ** 2)
mask = np.ones((H, W), dtype=np.float32)
# 中心からの四隅までの距離rに対して0.5r以内のものだけを通す(１になる)マスク
mask[r > np.sqrt((W//2)**2 + (H//2)**2) * p] = 0


# In[38]:


# IDFT with low-pass filter
out = np.zeros((H, W), dtype=np.float32)
x, y = np.meshgrid(np.arange(W), np.arange(H))

# filtering
trans_G *= mask
G[H//2:, W//2:] = trans_G[:H//2, :W//2]
G[H//2:, :W//2] = trans_G[:H//2, W//2:]
G[:H//2, W//2:] = trans_G[H//2:, :W//2]
G[:H//2, :W//2] = trans_G[H//2:, W//2:]

for h in range(H):
    for w in range(W):
        out[h, w] = np.abs(np.sum( G * np.exp( 2j * np.pi * (x * w / W + y * h / H)))) / np.sqrt(H * W)
        
out[out > 255] = 255
out = out.astype(np.uint8)


# In[39]:


result(out, "33")

