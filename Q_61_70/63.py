
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("gazo.png").astype(np.float32)
H, W, C = img.shape


# In[10]:


out = np.zeros((H, W), dtype=np.int)
out[img[:,:,0] > 0] = 1

flag = True
while(flag):
    flag = False
    tmp = out.copy()
    for y in range(H):
        for x in range(W):
            if out[y, x] == 0:
                continue
            else:
                x1 = tmp[y, min(x+1, W-1)]
                x2 = tmp[max(0, y-1), min(x+1, W-1)]
                x3 = tmp[max(0, y-1), x]
                x4 = tmp[max(0, y-1), max(0, x-1)]
                x5 = tmp[y, max(0, x-1)]
                x6 = tmp[min(y+1, H-1), max(0, x-1)]
                x7 = tmp[min(y+1, H-1), x]
                x8 = tmp[min(y+1, H-1), min(x+1, W-1)]
                cond1 = ((x1 + x3 + x5 + x7) < 4)
                cond2 = (((x1 - x1*x2*x3) + (x3 - x3*x4*x5) + (x5 - x5*x6*x7) + (x7 -x7*x8*x1)) == 1)
                cond3 = ((x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8) >= 3)
                if cond1 and cond2 and cond3:
                    flag = True
                    out[y, x] = 0
                
out = out.astype(np.uint8) * 255


# In[11]:


result(out, "63")

