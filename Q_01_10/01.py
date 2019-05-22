import cv2
img = cv2.imread("imori.jpg")
b = img[:, :, 0].copy()
g = img[:, :, 1].copy()
r = img[:, :, 2].copy()

# from RGB to BGR
img[:, :, 0] = r
img[:, :, 2] = b

cv2.imwrite("01.jpg", img)
cv2.imshow("01", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
