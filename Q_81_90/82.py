
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale

img = cv2.imread("thorino.jpg").astype(np.float32)
H, W, C = img.shape
b = img[:,:,0].copy()
g = img[:,:,1].copy()
r = img[:,:,2].copy()
gray_img = rgb2grayscale(r,g,b)

# Sobel filter
sobel_v = np.array([ [1, 2, 1], [0, 0, 0], [-1, -2, -1] ], dtype=np.float32)
sobel_h = np.array([ [1, 0, -1], [2, 0, -2], [1, 0, -1] ], dtype=np.float32)

tmp = np.pad(gray_img, (1, 1), 'edge')

Ix = np.zeros_like(gray_img, dtype=np.float32)
Iy = np.zeros_like(gray_img, dtype=np.float32)

for y in range(H):
    for x in range(W):
        Ix[y, x] = np.sum(tmp[y:y+3, x:x+3] * sobel_h)
        Iy[y, x] = np.sum(tmp[y:y+3, x:x+3] * sobel_v)

Ix2 = Ix**2
Iy2 = Iy**2
Ixy = Ix * Iy


# In[7]:


import cv2
import numpy as np
from save import result

# Read image
img = cv2.imread("thorino.jpg").astype(np.float32)
H, W, C = img.shape

## Grayscale
gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]

# Harris

## Sobel
sobely = np.array(((1, 2, 1),
                   (0, 0, 0),
                   (-1, -2, -1)), dtype=np.float32)

sobelx = np.array(((1, 0, -1),
                   (2, 0, -2),
                   (1, 0, -1)), dtype=np.float32)

tmp = np.pad(gray, (1, 1), 'edge')

Ix = np.zeros_like(gray, dtype=np.float32)
Iy = np.zeros_like(gray, dtype=np.float32)

for y in range(H):
    for x in range(W):
        Ix[y, x] = np.sum(tmp[y:y+3, x:x+3] * sobelx)
        Iy[y, x] = np.sum(tmp[y:y+3, x:x+3] * sobely)

Ix2 = Ix ** 2
Iy2 = Iy ** 2
Ixy = Ix * Iy

## gaussian
K_size = 3
sigma = 3
Ix2_t = np.pad(Ix2, (K_size // 2, K_size // 2), 'edge')
Iy2_t = np.pad(Iy2, (K_size // 2, K_size // 2), 'edge')
Ixy_t = np.pad(Ixy, (K_size // 2, K_size // 2), 'edge')

K = np.zeros((K_size, K_size), dtype=np.float)
for x in range(K_size):
    for y in range(K_size):
        _x = x - K_size // 2
        _y = y - K_size // 2
        K[y, x] = np.exp( -(_x**2 + _y**2) / (2 * (sigma**2)))
K /= (sigma * np.sqrt(2 * np.pi))
K /= K.sum()

for y in range(H):
    for x in range(W):
        Ix2[y,x] = np.sum(Ix2_t[y:y+K_size, x:x+K_size] * K)
        Iy2[y,x] = np.sum(Iy2_t[y:y+K_size, x:x+K_size] * K)
        Ixy[y,x] = np.sum(Ixy_t[y:y+K_size, x:x+K_size] * K)
        
Ix2 = Ix2.astype(np.uint8)
Iy2 = Iy2.astype(np.uint8)
Ixy = Ixy.astype(np.uint8)


# In[8]:


result(Ix2, "82_x")
result(Iy2, "82_y")
result(Ixy, "82_xy")

