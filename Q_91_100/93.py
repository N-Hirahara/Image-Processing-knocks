
# coding: utf-8

# In[5]:


import numpy as np

a = np.array((50, 50, 150, 150), dtype=np.float32)
b = np.array((60, 60, 170, 160), dtype=np.float32)

def IoU(a, b):
    W_r1, H_r1 = a[2]-a[0], a[3]-a[1]
    W_r2, H_r2 = b[2]-b[0], b[3]-b[1]
    r1 = (W_r1) * (H_r1)
    r2 = (W_r2) * (H_r2)
    
    dw = ((W_r1 + W_r2) - (abs(b[0]-a[0]) + abs(b[2]-a[2]))) / 2 
    if dw <= 0 : 
        dw = 0
    dh = ((H_r1 + H_r2) - (abs(b[1]-a[1]) + abs(b[3]-a[3]))) / 2 
    if dh <= 0 : 
        dh = 0
    rol = dw * dh
    
    return rol / (r1 + r2 - rol)


# In[6]:


print(IoU(a,b))

