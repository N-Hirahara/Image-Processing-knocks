
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

gray_img = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]

def bilinear(img, a):
    H, W = img.shape
    new_H = int(H * a)
    new_W = int(W * a)
    
    # 拡大画像の各座標
    new_x, new_y = np.meshgrid(np.arange(new_W), np.arange(new_H))
    # 拡大画像の各座標に対応する元画像上の座標
    orig_x = new_x / a
    orig_y = new_y / a
    
    ix = np.floor(orig_x).astype(np.int)
    iy = np.floor(orig_y).astype(np.int)
    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)
    
    dx = orig_x - ix
    dy = orig_y - iy
    
    out = (1-dx)*(1-dy)*img[iy,ix] + dx*(1-dy)*img[iy,ix+1] + (1-dx)*dy*img[iy+1,ix] + dx*dy*img[iy+1,ix+1]
    out[out > 255] = 255
    
    return out


# In[6]:


pyramid = [gray_img.astype(np.uint8)]
for i in range(1, 6):
    a = 2 ** i
    img = bilinear(gray_img, 1. / a).astype(np.uint8)
    pyramid.append(img)


# In[8]:


for i, img in enumerate(pyramid):
    result(img, "75-{}".format(i))

