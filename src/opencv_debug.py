import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from imutils import opencv2matplotlib

from opencv_utils import cvt_single_color


def show(image: npt.NDArray, contours=None):
    """
    Show image for debugging.

    :param image: The image to show.
    :param contours: Optional contours.
    """
    if contours is None:
        to_show = image
    else:
        to_show = image.copy()
        cv2.drawContours(
            to_show, contours, -1, _get_contour_color(image), image.shape[0] // 100
        )
    plt.figure()
    plt.imshow(opencv2matplotlib(to_show))


def _get_contour_color(image: npt.NDArray) -> tuple[int, int, int]:
    """
    Get a proper color for the contour.

    :param image: The image.
    :return: A BGR color for the contour.
    """
    # get hsv of center pixel
    h, w = image.shape[:2]
    center = image[h // 2, w // 2]
    hsv = cvt_single_color(center, cv2.COLOR_BGR2HSV)

    # use red contour for green image
    if np.all((40, 40, 40) <= hsv) and np.all(hsv <= (70, 255, 255)):
        return 0, 0, 255
    # green for others
    return 0, 255, 0
