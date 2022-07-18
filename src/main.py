import cv2
import imutils
import numpy.typing as npt


def find_screen(photo: npt.NDArray) -> npt.NDArray:
    """
    Find the screen in the photo.

    :param photo: A photo of the screen. White screen in dark background is expected.
    :return: Four points of the screen.
    """
    # get the contours
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
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
    assert len(approx) == 4

    return approx[:, 0, :]
