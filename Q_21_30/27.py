
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float)

H, W, C = img.shape


# In[79]:


# Bi-cubic interpolation
a = 1.5
aH = int(a * H)
aW = int(a * W)

# 拡大画像の座標配列
new_x, new_y = np.meshgrid(np.arange(aW), np.arange(aH))
# 拡大画像の座標に対応する元画像の座標配列(float)
orig_x = new_x / a
orig_y = new_y / a

# ix, iy ： orig_x, orig_y に最も近い左上の点の座標
ix = np.floor(orig_x).astype(np.int)
iy = np.floor(orig_y).astype(np.int)
ix = np.minimum(ix, W-1)
iy = np.minimum(iy, H-1)

dx1 = orig_x - (ix -1)
dx2 = orig_x - ix
dx3 = np.abs(orig_x - (ix+1))
dx4 = np.abs(orig_x - (ix+2))
dy1 = orig_y - (iy -1)
dy2 = orig_y - iy
dy3 = np.abs(orig_y - (iy+1))
dy4 = np.abs(orig_y - (iy+2))

dxs = [dx1, dx2, dx3, dx4]
dys = [dy1, dy2, dy3, dy4]


# In[80]:


def weight(t):
    a = -1
    w = np.zeros_like(t)
    t = np.abs(t)
    idx = np.where(t <= 1)
    w[idx] = ((a+2)*(t**3) - (a+3)*(t**2) + 1)[idx]
    idx = np.where((1 < t) & (t <= 2))
    w[idx] = (a*(t**3) - 5 * a * (t**2) + 8*a*t - 4*a)[idx]
    idx = np.where(2 < t)
    w[idx] = 0
    return w


# In[81]:


wxy_sum = np.zeros((aH, aW, C), dtype=np.float32)
out = np.zeros((aH, aW, C), dtype=np.float32)

for j in range(-1, 3):
    for i in range(-1, 3):
        idx_x = np.minimum(np.maximum((ix + i), 0), W-1)
        idx_y = np.minimum(np.maximum((iy + j), 0), H-1)
        
        wx = weight(dxs[i + 1])
        wy = weight(dys[j + 1])
        # expand to 3-dimension
        wx = np.repeat(np.expand_dims(wx, axis=-1), 3, axis=-1)
        wy = np.repeat(np.expand_dims(wy, axis=-1), 3, axis=-1)

        wxy_sum += wx * wy
        out += img[idx_y, idx_x] * wx * wy

out /= wxy_sum
out[out > 255] = 255
# out[out < 0] = 0
out = out.astype(np.uint8)


# In[82]:


result(out, "27")

