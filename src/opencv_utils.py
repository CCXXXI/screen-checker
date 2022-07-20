import cv2
import numpy as np


def cvt_single_color(src, code):
    """
    Single color conversion.

    :param src: The source color.
    :param code: Something like cv2.COLOR_BGR2HSV.
    :return: The converted color.
    """
    return cv2.cvtColor(np.uint8([[src]]), code)[0][0]
