
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float)
H, W, C = img.shape


# In[11]:


# Bi-cubic
a = 1.5
aH = int(a * H)
aW = int(a * W)

new_x, new_y = np.meshgrid(np.arange(aW), np.arange(aH))
# 拡大画像の各座標に対応する、元画像の座標配列(float)
orig_x = new_x / a
orig_y = new_y / a


# In[16]:


ix = np.floor(orig_x).astype(np.int)
iy = np.floor(orig_y).astype(np.int)
ix = np.minimum(ix, W-1)
iy = np.minimum(iy, H-1)


# In[33]:


dx1 = orig_x - (ix-1)
dx2 = orig_x - ix
dx3 = (ix+1) - orig_x
dx4 = (ix+2) - orig_x
dy1 = orig_y - (iy-1)
dy2 = orig_y - iy
dy3 = (iy+1) - orig_y
dy4 = (iy+2) - orig_y

dxs = [dx1, dx2, dx3, dx4]
dys = [dy1, dy2, dy3, dy4]


# In[34]:


def weight(t):
    a = -1
    w = np.zeros_like(t)
    t = np.abs(t)
    idx = np.where(t <= 1)
    w[idx] = ((a+2)*(t**3) - (a+3)*(t**2) + 1)[idx]
    idx = np.where((1 < t) & (t <= 2))
    w[idx] = (a*(t**3) - 5*a*(t**2) + 8*a*t - 4*a)[idx]
    idx = np.where(2 < t)
    w[idx] = 0
    return w


# In[38]:


wxy_sum = np.zeros((aH, aW, C), dtype=np.float32)
out = np.zeros((aH, aW, C), dtype=np.float32)

for j in range(-1, 3):
    for i in range(-1, 3):
        idx_x = np.minimum(np.maximum(ix + i, 0) , W-1)
        idx_y = np.minimum(np.maximum(iy + i, 0) , H-1)
        
        wx = weight(dxs[i + 1])
        wy = weight(dys[j + 1])
        wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
        wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)
        
        wxy_sum += wx * wy
        out += img[idx_y, idx_x] * wx * wy

out /= wxy_sum
out[out > 255] = 255
out = out.astype(np.uint8)


# In[39]:


result(out, "re_27")

