import cv2
import numpy as np
from pytest import mark

from opencv_utils import cvt_single_color


@mark.parametrize(
    "src, code, expected",
    (
        ((0, 128, 255), cv2.COLOR_BGR2RGB, (255, 128, 0)),
        ((0, 128, 255), cv2.COLOR_BGR2HSV, (15, 255, 255)),
    ),
)
def test_cvt_single_color(src, code, expected):
    assert np.array_equal(cvt_single_color(src, code), expected)
