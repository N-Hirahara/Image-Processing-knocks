
# coding: utf-8

# In[7]:


import cv2
import numpy as np
from save import result

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


def hog(gray_img):
    H, W = gray_img.shape
    gray_img = np.pad(gray_img, (1,1), 'edge')
        
    # gradients
    gx = gray_img[1:H+1, 2:] - gray_img[1:H+1, :W]
    gy = gray_img[2:, 1:W+1] - gray_img[:H, 1:W+1]
    gx[gx == 0] = 0.000001

    # magnitude and angle of gradients
    mag = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    ang = np.arctan(gy / gx) + np.pi/2

    # mag = (mag / np.max(mag) * 255).astype(np.uint8)
    # quantize the angle of gradients
    q_ang = np.zeros_like(ang, dtype=np.int)
    d = np.pi / 9
    for i in range(9):
        q_ang[np.where((ang >= d * i) & (ang <= d * (i+1)))] = i

    # cell
    N = 8
    cell_H = H // N
    cell_W = W // N
    hist = np.zeros((cell_H, cell_W, 9), dtype=np.float32)

    for h in range(cell_H):
        for w in range(cell_W):
            for y in range(N):
                for x in range(N):
                    hist[h, w, q_ang[N*h+y, N*w+x]] += mag[N*h+y, N*w+x]

    # normalize the histogram
    C = 3
    c = C // 2
    epsilon = 1
    for h in range(cell_H):
        for w in range(cell_W):
            hist[h, w] /= np.sqrt(np.sum(np.power(hist[max(0, h-c):min(cell_H, h+C-c), max(0, w-c):min(cell_W, w+C-c)], 2)) + epsilon)

    return hist

def resize_bilinear(img, h, w):
    H_, W_ = img.shape
    
    # 拡大画像の各座標
    new_x, new_y = np.meshgrid(np.arange(w), np.arange(h))
    # 拡大画像の各座標に対応する元画像上の座標
    orig_x = new_x / (1. * w / W_)
    orig_y = new_y / (1. * h / H_)
    
    ix = np.floor(orig_x).astype(np.int)
    iy = np.floor(orig_y).astype(np.int)
    ix = np.minimum(ix, W_-2)
    iy = np.minimum(iy, H_-2)
    
    dx = orig_x - ix
    dy = orig_y - iy
    
    out = (1-dx)*(1-dy)*img[iy,ix] + dx*(1-dy)*img[iy,ix+1] + (1-dx)*dy*img[iy+1,ix] + dx*dy*img[iy+1,ix+1]
    out[out > 255] = 255
    
    return out


# In[2]:


class NN:
    def __init__(self, ind=2, w=64, outd=1, lr=0.1):
        self.w1 = np.random.normal(0, 1, [ind, w])
        self.b1 = np.random.normal(0, 1, [w])
        self.w2 = np.random.normal(0, 1, [w, w])
        self.b2 = np.random.normal(0, 1, [w])
        self.wout = np.random.normal(0, 1, [w, outd])
        self.bout = np.random.normal(0, 1, [outd])
        self.lr = lr

    def forward(self, x):
        self.z1 = x
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        grad_En = En #np.array([En for _ in range(t.shape[0])])
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        self.wout -= self.lr * grad_wout#np.expand_dims(grad_wout, axis=-1)
        self.bout -= self.lr * grad_bout

        # backpropagation 2nd inter layer (added layer)
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        
        # backpropagation 1st inter layer
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# train NN
img = cv2.imread("imori_1.jpg").astype(np.float32)
H, W, C = img.shape

gray_img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

gt = np.array((47, 41, 129, 103), dtype=np.float32)

# cropping and make database
np.random.seed(1)
crop_size = 60
crop_num = 200
resize_len = 32
F_n = ((resize_len // 8) ** 2) * 9

db = np.zeros((crop_num, F_n+1))

for i in range(crop_num):
    x1 = np.random.randint(W-60)
    y1 = np.random.randint(H-60)
    cropping = np.array((x1, y1, x1+crop_size, y1+crop_size), dtype=np.float32)
    label = 1 if IoU(cropping, gt) >= 0.5 else 0
    
    cropped = gray_img[y1:y1+crop_size, x1:x1+crop_size]
    cropped = resize_bilinear(cropped, resize_len, resize_len)

    db[i, :F_n] = hog(cropped).ravel()
    db[i, -1] = label

# train NN
nn = NN(ind=F_n, lr=0.01)
num_epoch = 10000
for i in range(num_epoch):
    nn.forward(db[:, :F_n])
    nn.train(db[:, :F_n], db[:, -1].reshape(-1, 1))


# In[17]:


# target image
img = cv2.imread("imori_many.jpg").astype(np.float32)
H, W, C = img.shape

gray_img = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

# resize
resize_len = 32
# HOG features
H_feature = ((resize_len // 8) ** 2) * 9

# bounding boxes [h, w]
recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)

detected = []

# sliding window
for y in range(0, H, 4):
    for x in range(0, W, 4):
        for rec in recs:
            dh, dw = int(rec[0]//2), int(rec[1]//2)
            x1 = max(0, x-dw)
            x2 = min(W, x+dw)
            y1 = max(0, y-dh)
            y2 = min(H, y+dh)
            focus_region = gray_img[y1:y2, x1:x2]
            focus_region = resize_bilinear(focus_region, resize_len, resize_len)
            focus_hog = hog(focus_region).ravel()
            
            proba = nn.forward(focus_hog).item()
            if proba >= 0.65:
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255))
                detected.append([x1, y1, x2, y2, proba])


# In[19]:


detected


# In[20]:


out = img.astype(np.uint8)
result(out, "98")

