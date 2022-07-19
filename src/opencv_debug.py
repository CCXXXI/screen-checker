import cv2
import matplotlib.pyplot as plt
import numpy.typing as npt
from imutils import opencv2matplotlib


def show(image: npt.NDArray, contours=None):
    """
    Show image for debugging.

    :param image: The image to show.
    :param contours: Optional contours.
    """
    tmp = image.copy()
    if contours is not None:
        cv2.drawContours(tmp, contours, -1, (0, 255, 0), 16)
    plt.imshow(opencv2matplotlib(tmp))
