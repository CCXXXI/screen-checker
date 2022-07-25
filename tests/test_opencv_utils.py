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
def test_cvt_single_color_uint8(src, code, expected):
    assert np.array_equal(cvt_single_color(src, code), expected)


@mark.parametrize(
    "src, code, expected",
    (
        ((0, 0.5, 1), cv2.COLOR_BGR2RGB, (1, 0.5, 0)),
        ((0, 0.5, 1), cv2.COLOR_BGR2HSV, (30, 1, 1)),
    ),
)
def test_cvt_single_color_float32(src, code, expected):
    assert np.allclose(cvt_single_color(src, code, np.float32), expected)
