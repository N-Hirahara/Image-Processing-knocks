
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("gazo.png").astype(np.float32)
H, W, C = img.shape


# In[32]:


out = np.zeros((H, W), dtype=np.int)
out[img[:,:,0] > 0] = 1

out = 1 - out
flag = True
while flag:
    flag = False
    step1 = []
    for y in range(H):
        for x in range(W):
            x1 = out[y,x]
            x2 = out[max(y-1, 0), x]
            x3 = out[max(y-1, 0), min(x+1, W-1)]
            x4 = out[y, min(x+1, W-1)]
            x5 = out[min(y+1, H-1), min(x+1, W-1)]
            x6 = out[min(y+1, H-1), x]
            x7 = out[min(y+1, H-1), max(x-1, 0)]
            x8 = out[y, max(x-1, 0)]
            x9 = out[max(y-1, 0), max(x-1, 0)]

            # step1
            # condition1
            if x1 > 0:
                continue

            # condition2
            c2 = 0
            if (x3 - x2) == 1:
                c2 += 1
            if (x4 - x3) == 1:
                c2 += 1
            if (x5 - x4) == 1:
                c2 += 1
            if (x6 - x5) == 1:
                c2 += 1
            if (x7 - x6) == 1:
                c2 += 1
            if (x8 - x7) == 1:
                c2 += 1
            if (x9 - x8) == 1:
                c2 += 1
            if (x2 - x9) == 1:
                c2 += 1
            if c2 != 1:
                continue

            # condition 3
            if (x2+x3+x4+x5+x6+x7+x8+x9) < 2 or (x2+x3+x4+x5+x6+x7+x8+x9) > 6:
                continue

            # condition 4
            if x2 == 0 and x4 == 0 and x6 == 0:
                continue

            # condition 5
            if x4 == 0 and x6 == 0 and x8 == 0:
                continue

            flag = True
            step1.append([y, x])

    for y, x in step1:
        out[y, x] = 1

    step2 = []
    for y in range(H):
        for x in range(W):
            x1 = out[y,x]
            x2 = out[max(y-1, 0), x]
            x3 = out[max(y-1, 0), min(x+1, W-1)]
            x4 = out[y, min(x+1, W-1)]
            x5 = out[min(y+1, H-1), min(x+1, W-1)]
            x6 = out[min(y+1, H-1), x]
            x7 = out[min(y+1, H-1), max(x-1, 0)]
            x8 = out[y, max(x-1, 0)]
            x9 = out[max(y-1, 0), max(x-1, 0)]

            # step2
            # condition1
            if x1 > 0:
                continue

            # condition2
            c2 = 0
            if (x3 - x2) == 1:
                c2 += 1
            if (x4 - x3) == 1:
                c2 += 1
            if (x5 - x4) == 1:
                c2 += 1
            if (x6 - x5) == 1:
                c2 += 1
            if (x7 - x6) == 1:
                c2 += 1
            if (x8 - x7) == 1:
                c2 += 1
            if (x9 - x8) == 1:
                c2 += 1
            if (x2 - x9) == 1:
                c2 += 1
            if c2 != 1:
                continue

            # condition 3
            if (x2+x3+x4+x5+x6+x7+x8+x9) < 2 or (x2+x3+x4+x5+x6+x7+x8+x9) > 6:
                continue

            # condition 4
            if x2 == 0 and x4 == 0 and x8 == 0:
                continue

            # condition 5
            if x2 == 0 and x6 == 0 and x8 == 0:
                continue

            flag = True
            step2.append([y, x])

            
    for y, x in step2:
        out[y, x] = 1


out = 1 - out
out = out.astype(np.uint8)*255


# In[33]:


result(out, "65")

