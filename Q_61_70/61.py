
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("renketsu.png").astype(np.float32)
H, W, C = img.shape


# In[4]:


tmp = np.zeros((H, W), dtype=np.int)
tmp[img[:,:,0] > 0] = 1

out = np.zeros_like(img, dtype=np.uint8)

for y in range(H):
    for x in range(W):
        if tmp[y, x] < 1:
            continue
        
        x1 = tmp[y, min(x+1, W-1)]
        x2 = tmp[max(0, y-1), min(x+1, W-1)]
        x3 = tmp[max(0, y-1), x]
        x4 = tmp[max(0, y-1), max(0, x-1)]
        x5 = tmp[y, max(0, x-1)]
        x6 = tmp[min(y+1, H-1), max(0, x-1)]
        x7 = tmp[min(y+1, H-1), x]
        x8 = tmp[min(y+1, H-1), min(x+1, W-1)]
        
        s = 0
        s += (x1 - x1*x2*x3) + (x3 - x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)
        
        if s == 0:
            out[y,x] = [0, 0, 255]
        elif s == 1:
            out[y,x] = [0, 255, 0]
        elif s == 2:
            out[y,x] = [255, 0, 0]
        elif s == 3:
            out[y,x] = [0, 255, 255]
        elif s == 4:
            out[y,x] = [255, 0, 255]
            
out = out.astype(np.uint8)


# In[5]:


result(out, "61")

