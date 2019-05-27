
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from save import result
from filtering import rgb2grayscale

img = cv2.imread("imori.jpg").astype(np.float32)
H, W, C = img.shape

b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r= img[:, :, 2].copy()

def RGB2YCbCr(r, g, b):
    Y = 0.299 * r + 0.5870 * g + 0.114 * b
    Cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    Cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128
    return Y, Cb, Cr

def YCbCr2RGB(y, cb, cr):
    R = y + (cr - 128) * 1.402
    G = y - (cb - 128) * 0.3441 - (cr - 128) * 0.7139
    B = y + (cb - 128) * 1.7718
    return R, G, B

Y, Cb, Cr = RGB2YCbCr(r, g, b)


# In[2]:


# Discrete cosine transformation
T = 8
K = 8
F_y = np.zeros_like(Y, dtype=np.float64)
F_cb = np.zeros_like(Cb, dtype=np.float64)
F_cr = np.zeros_like(Cr, dtype=np.float64)

def weight(x, y, u, v):
    if u==0:
        cu = 1 / np.sqrt(2)
    else:
        cu = 1
    if v == 0:
        cv = 1 / np.sqrt(2)
    else:
        cv = 1
    return (2 * cu * cv / T) * np.cos((2*x+1) * u * np.pi / (2*T)) * np.cos((2*y+1) * v * np.pi / (2*T))

#  quantization table for Y
Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
               (12, 12, 14, 19, 26, 58, 60, 55),
               (14, 13, 16, 24, 40, 57, 69, 56),
               (14, 17, 22, 29, 51, 87, 80, 62),
               (18, 22, 37, 56, 68, 109, 103, 77),
               (24, 35, 55, 64, 81, 104, 113, 92),
               (49, 64, 78, 87, 103, 121, 120, 101),
               (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)

# quantization table for Cb, Cr
Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
               (18, 21, 26, 66, 99, 99, 99, 99),
               (24, 26, 56, 99, 99, 99, 99, 99),
               (47, 66, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99),
               (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)


# In[3]:


for yi in range(0, H, T):
    for xi in range(0, W, T):
        for v in range(T):
            for u in range(T):
                for y in range(T):
                    for x in range(T):
                        F_y[yi+v, xi +u] += Y[yi+y, xi+x] * weight(x, y, u, v)
                        F_cb[yi+v, xi +u] += Cb[yi+y, xi+x] * weight(x, y, u, v)
                        F_cr[yi+v, xi +u] += Cr[yi+y, xi+x] * weight(x, y, u, v)
        F_y[yi:yi+T, xi:xi+T] = np.round(F_y[yi:yi+T, xi:xi+T] / Q1) * Q1
        F_cb[yi:yi+T, xi:xi+T] = np.round(F_cb[yi:yi+T, xi:xi+T] / Q2) * Q2
        F_cr[yi:yi+T, xi:xi+T] = np.round(F_cr[yi:yi+T, xi:xi+T] / Q2) * Q2


# In[4]:


# IDCT
out_y = np.zeros((H, W), dtype=np.float64)
out_cb = np.zeros((H, W), dtype=np.float64)
out_cr = np.zeros((H, W), dtype=np.float64)

for yi in range(0, H, T):
    for xi in range(0, W, T):
        for y in range(T):
            for x in range(T):
                for v in range(K):
                    for u in range(K):
                        out_y[yi+y, xi +x] += F_y[yi+v, xi+u] * weight(x, y, u, v)
                        out_cb[yi+y, xi +x] += F_cb[yi+v, xi+u] * weight(x, y, u, v)
                        out_cr[yi+y, xi +x] += F_cr[yi+v, xi+u] * weight(x, y, u, v)

out_y[out_y > 255] = 255
out_cb[out_cb > 255] = 255
out_cr[out_cr > 255] = 255


# In[5]:


R, G, B = YCbCr2RGB(out_y, out_cb, out_cr)

out = np.zeros_like(img, dtype=np.float32)
out[:, :, 0] = B
out[:, :, 1] = G
out[:, :, 2] = R
out = out.astype(np.uint8)


# In[6]:


result(out, "40")

