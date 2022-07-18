from enum import Enum

import cv2
import imutils
import numpy.typing as npt


class Color(Enum):
    """Colors."""

    BLUE = 0
    GREEN = 1
    RED = 2
    WHITE = 3


def find_screen(photo: npt.NDArray, color: Color = Color.WHITE) -> npt.NDArray:
    """
    Find the screen in the photo.

    :param photo: A photo of the screen.
    :param color: The color of the screen.
    :return: Four (x, y) points which are the four corners of the screen.
    """
    # BGR to gray
    if color is Color.WHITE:
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    else:
        gray = photo[..., color.value]

    # get the contours
    binary = cv2.threshold(gray, None, 255, cv2.THRESH_OTSU)[1]
    contours = imutils.grab_contours(
        cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    )

    # the contour of the screen should be the largest one
    screen_contour = max(contours, key=cv2.contourArea)

    # it should be a quadrilateral
    approx = cv2.approxPolyDP(
        screen_contour, 0.1 * cv2.arcLength(screen_contour, True), True
    )
    if len(approx) != 4:
        raise ValueError("Cannot find the screen.")

    return approx[:, 0, :]
