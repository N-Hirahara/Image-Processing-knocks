
# coding: utf-8

# In[1]:


import cv2 
import numpy as np


# In[28]:


img = cv2.imread("imori.jpg").astype(np.float32)
scaled_img = img / 255


# RGB to HSV
max_v = np.max(scaled_img, axis=2)
min_v = np.min(scaled_img, axis=2)
min_bgr = np.argmin(scaled_img, axis=2)

# Hue
H = np.zeros_like(max_v)
H[np.where(max_v == min_v)] = 0
# if min_bgr = B
idx = np.where(min_bgr == 0)
H[idx] = 60 * (scaled_img[:, :, 1][idx] - scaled_img[:, :, 2][idx]) / (max_v[idx] - min_v[idx]) + 60
# if min_bgr = R
idx = np.where(min_bgr == 2)
H[idx] = 60 * (scaled_img[:, :, 0][idx] - scaled_img[:, :, 1][idx]) / (max_v[idx] - min_v[idx]) + 180
# if min_bgr = G
idx = np.where(min_bgr == 1)
H[idx] = 60 * (scaled_img[:, :, 2][idx] - scaled_img[:, :, 0][idx]) / (max_v[idx] - min_v[idx]) + 300

# Saturation
S = max_v.copy() - min_v.copy()

# Value
V = max_v.copy()


# In[32]:


# transpose hue
H = (H + 180) % 360

# HSV to RGB
out = np.zeros_like(img)

C = S
H_ = H / 60
X = C * (1 - np.abs((H_ % 2) - 1))

Zero = np.zeros_like(H)
bgr_values = [
    [Zero, X, C],
    [Zero, C, X],
    [X, C, Zero],
    [C, X, Zero],
    [C, Zero, X],
    [X, Zero, C]
]

for i in range(6):
    idx = np.where((i <= H_) & (H_ < (i+1)))
    out[:, :, 0][idx] = (V-C)[idx] + bgr_values[i][0][idx]
    out[:, :, 1][idx] = (V-C)[idx] + bgr_values[i][1][idx]
    out[:, :, 2][idx] = (V-C)[idx] + bgr_values[i][2][idx]

# rescaled
out = (out * 255).astype(np.uint8)


# In[31]:


cv2.imwrite("05.jpg", out)
cv2.imshow("05", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

