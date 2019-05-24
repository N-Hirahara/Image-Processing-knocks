
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float)


# In[31]:


H, W, C = img.shape

# Bi-linear interpolation
a = 1.5
new_H = int(a * H)
new_W = int(a * W)

y = np.arange(new_H).repeat(new_W).reshape(new_H, -1)
x = np.tile(np.arange(new_W), (new_H, 1))
y = y / a
x = x / a

ix = np.floor(x).astype(np.int)
iy = np.floor(y).astype(np.int)
# 画像の右端、下端の処理はこれで対応しているっぽい？
ix = np.minimum(ix, W-2)
iy = np.minimum(iy, H-2)

dx = x - ix
dy = y - iy
# ３次元チャンネル分に拡張
dx = np.repeat(np.expand_dims(dx, axis=-1), 3, axis=-1)
dy = np.repeat(np.expand_dims(dy, axis=-1), 3, axis=-1)

out = (1-dx)*(1-dy)*img[iy,ix] + dx*(1-dy)*img[iy,ix+1] + (1-dx)*dy*img[iy+1,ix] + dx*dy*img[iy+1,ix+1]

out[out > 255] = 255
out = out.astype(np.uint8)


# In[32]:


result(out, "26")

