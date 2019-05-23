import cv2

def result(img, name):
    cv2.imwrite(name+".jpg", img)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()