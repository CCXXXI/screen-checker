import cv2
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
    plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
