import cv2
from pytest import mark, raises

from main import find_screen, Color


@mark.parametrize(
    "file, color",
    (
        *(("../resources/white 0.jpg", c) for c in Color),
        ("../resources/blue 0.jpg", Color.BLUE),
        ("../resources/green 0.jpg", Color.GREEN),
        ("../resources/red 0.jpg", Color.RED),
    ),
)
def test_find_screen(file: str, color: Color):
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
        *(("../resources/blue 0.jpg", c) for c in (Color.GREEN, Color.RED)),
        *(("../resources/green 0.jpg", c) for c in (Color.BLUE, Color.RED)),
        *(("../resources/red 0.jpg", c) for c in (Color.BLUE, Color.GREEN)),
        ("../resources/red 0.jpg", Color.BLACK),
    ),
)
def test_find_screen_failed(file: str, color: Color):
    img = cv2.imread(file)
    assert img is not None

    with raises(ValueError):
        find_screen(img, color)
