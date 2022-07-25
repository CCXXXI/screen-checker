import cv2
import numpy as np


def cvt_single_color(src, code, np_type=np.uint8):
    """
    Single color conversion.

    :param src: The source color.
    :param code: Something like cv2.COLOR_BGR2HSV.
    :param np_type: np.uint8 or np.float32 or something else.
    :return: The converted color.
    """
    return cv2.cvtColor(np_type([[src]]), code)[0][0]
