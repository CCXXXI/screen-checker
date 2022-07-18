import cv2
import imutils
import matplotlib.pyplot as plt

from data_types import Image


def show(image: Image, contours=None):
    """
    Show image for debugging.

    :param image: The image to show.
    :param contours: Optional contours.
    """
    tmp = image.copy()
    if contours is not None:
        cv2.drawContours(tmp, contours, -1, (0, 255, 0), 16)
    plt.imshow(imutils.opencv2matplotlib(tmp))
