import cv2
from pytest import mark

from main import find_screen


@mark.parametrize("file", ["../resources/white 0.jpg"])
def test_find_screen(file: str):
    img = cv2.imread(file)
    assert img is not None

    res = find_screen(img)
    assert len(res) == 4
    for point in res:
        assert len(point) == 2
        assert all(point > 0)
