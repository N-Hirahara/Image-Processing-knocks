
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result

img = cv2.imread("seg.png").astype(np.float32)
H, W, C = img.shape

# initialize label
label = np.zeros((H, W), dtype=np.int)
label[img[:,:,0] > 0] = 1

# look up table, labels: 0 ~ N
lookup_table = [0] * (H*W)
label_num = 0

for y in range(H):
    for x in range(W):
        if label[y, x] == 0:
            continue
        # 8 neighbors
        up_left = label[max(y-1, 0), max(x-1, 0)]
        up = label[max(y-1, 0), x]
        up_right = label[max(y-1, 0), min(x+1, W-1)]
        left = label[y, max(x-1, 0)]
        if up_left == 0 and up==0 and up_right==0 and left == 0:
            label_num += 1
            label[y, x] = label_num
        else:
            neighbors = [up_left, up, up_right, left]
            label_nums = [a for a in neighbors if a > 0]
            min_label = min(label_nums)
            label[y, x] = min_label
            for num in label_nums:
                if lookup_table[num] != 0:
                    min_label = min(min_label, lookup_table[num])
            for num in label_nums:
                lookup_table[num] = min_label

# update labels based on look up table
new_label = 0
for label_tag in range(1, label_num+1):
    flag = True
    for i in range(label_num+1):
        if lookup_table[i] == label_tag:
            if flag:
                new_label += 1
                flag = False
            lookup_table[i] = new_label


# In[2]:


lookup_table


# In[3]:


label_colors = [[0,0,255], [0,255,0], [255, 0, 0]]
out = np.zeros_like(img, dtype=np.uint8)

for i, lookup in enumerate(lookup_table[1:]):
    out[label == (i+1)] = label_colors[lookup-1]


# In[4]:


result(out, "59")

