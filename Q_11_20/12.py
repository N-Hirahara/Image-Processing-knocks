
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("imori.jpg")
H, W, C = img.shape


# In[2]:


# motion filter
F_size = 3

# zero padding
pad = F_size // 2
out = np.zeros((H+pad*2, W+pad*2, C))
out[pad:pad+H, pad:pad+W, :] = img.copy()


# In[3]:


# kernel
K = np.zeros((F_size, F_size))
for i in range(F_size):
    K[i][i] = 1 / F_size

    
out_tmp = out.copy()

for h in range(H):
    for w in range(W):
        for c in range(C):
            out[pad+h, pad+w, c] = np.sum(K * out_tmp[h:h+F_size, w:w+F_size, c])

out = out[pad:pad+H, pad:pad+W].astype(np.uint8)


# In[4]:


result(out, "12")


# In[4]:


from filtering import filtering


# In[6]:


out2 = filtering(img, K, True)
cv2.imshow("",out2)
cv2.waitKey(0)
cv2.destroyAllWindows

