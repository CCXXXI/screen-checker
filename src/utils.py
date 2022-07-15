from matplotlib.pyplot import imshow
import cv2


def show(cv_img):
    imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
