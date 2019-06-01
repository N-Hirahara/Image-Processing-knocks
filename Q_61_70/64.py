
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("gazo.png").astype(np.float32)
H, W, C = img.shape


# In[2]:


def connect8(X):
    tmp = X.copy()
    for i, neighbor in enumerate(X):
        tmp[i] = 1 - neighbor
    return (tmp[0] - tmp[0]*tmp[1]*tmp[2]) + (tmp[2] - tmp[2]*tmp[3]*tmp[4]) + (tmp[4] - tmp[4]*tmp[5]*tmp[6]) + (tmp[6] - tmp[6]*tmp[7]*tmp[0])


# In[6]:


out = np.zeros((H, W), dtype=np.int)
out[img[:,:,0] > 0] = 1

count = 0
flag = True
while(flag):
    flag = False
    
    for y in range(H):
        for x in range(W):
            tmp = out.copy()
            if out[y, x] < 1:
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
                X = [x1, x2, x3, x4, x5, x6, x7, x8] 

                # condition 1
                for neighbor in X[0::2]:
                    cond1= False
                    if neighbor == 0:
                        cond1 = True
                        break
                # condition 2
                cond2 = (connect8(X) == 1)
                # condition 3
                cond3 = (np.sum(np.abs(tmp[max(y-1,0):min(y+2,H), max(x-1,0):min(x+2,W)])) >= 3)
                # condition 4
                for neighbor in X:
                    cond4 = False
                    if neighbor == 1:
                        cond4 = True
                        break
                # condition 5-1
                cond5_1 = True
                for neighbor in X:
                    if neighbor == -1:
                        cond5_1 = False
                        break
                # condition 5-2
                cond5_2 = False
                for i, _ in enumerate(X):
                    X_ = X.copy()
                    c = 0
                    X_[i] = 0
                    if connect8(X_) == 1:
                        c += 1
                if c == 8:
                    cond5_2 = True
                
                cond5 = cond5_1 or cond5_2
                
                if cond1 and cond2 and cond3 and cond4 and cond5:
                    flag = True
                    out[y, x] = -1
                    
    out[np.where(out ==-1)] = 0

out = out.astype(np.uint8) * 255


# In[7]:


result(out, "64")

