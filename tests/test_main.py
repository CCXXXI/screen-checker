import cv2
from pytest import mark, raises

from main import find_screen


@mark.parametrize(
    "file, color",
    (
        *(("../resources/white 0.jpg", c) for c in (None, *range(3))),
        ("../resources/blue 0.jpg", 0),
        ("../resources/green 0.jpg", 1),
        ("../resources/red 0.jpg", 2),
    ),
)
def test_find_screen(file: str, color: int):
    img = cv2.imread(file)
    assert img is not None

    res = find_screen(img, color)
    assert len(res) == 4
    for point in res:
        assert len(point) == 2
        assert all(point > 0)


@mark.parametrize(
    "file, color",
    (
        *(("../resources/blue 0.jpg", c) for c in (1, 2)),
        *(("../resources/green 0.jpg", c) for c in (0, 2)),
        *(("../resources/red 0.jpg", c) for c in (0, 1)),
    ),
)
def test_find_screen_failed(file: str, color: int):
    img = cv2.imread(file)
    assert img is not None

    with raises(ValueError):
        find_screen(img, color)
