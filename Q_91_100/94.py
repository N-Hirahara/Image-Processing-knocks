
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

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


# In[2]:


img = cv2.imread("imori_1.jpg").astype(np.float32)
H, W, C = img.shape

# cropping
np.random.seed(1)
crop_size = 60
crop_num = 200


gt = np.array((47, 41, 129, 103), dtype=np.float32)

# label1:red, label0:blue, GT:green
img = cv2.rectangle(img, (gt[0], gt[1]), (gt[2],gt[3]), (0,255,0))
colors = [[255,0,0], [0,0,255]]
for i in range(crop_num):
    x1 = np.random.randint(W-60)
    y1 = np.random.randint(H-60)
    cropping = np.array((x1, y1, x1+crop_size, y1+crop_size), dtype=np.float32)
    label = 1 if IoU(cropping, gt) >= 0.5 else 0
    img = cv2.rectangle(img, (x1, y1), (x1+crop_size, y1+crop_size), colors[label])
    
out = img.astype(np.uint8)


# In[3]:


result(out, "94")

