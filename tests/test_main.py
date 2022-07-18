import cv2
from pytest import mark, raises

from main import Color, find_screen, check_screen


@mark.parametrize(
    "file, color",
    (
        *(
            ("../resources/white 0.jpg", c)
            for c in (Color.WHITE, Color.RED, Color.GREEN, Color.BLUE)
        ),
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
        ("../resources/white 0.jpg", Color.BLACK),
        *(("../resources/blue 0.jpg", c) for c in (Color.GREEN, Color.RED)),
        *(("../resources/green 0.jpg", c) for c in (Color.BLUE, Color.RED)),
        *(("../resources/red 0.jpg", c) for c in (Color.BLUE, Color.GREEN)),
    ),
)
def test_find_screen_failed(file: str, color: Color):
    img = cv2.imread(file)
    assert img is not None

    with raises(ValueError):
        find_screen(img, color)


@mark.parametrize(
    "file, color",
    (
        ("../resources/white 0.jpg", Color.WHITE),
        ("../resources/blue 0.jpg", Color.BLUE),
        ("../resources/green 0.jpg", Color.GREEN),
        ("../resources/red 0.jpg", Color.RED),
    ),
)
def test_check_screen_pass(file: str, color: Color):
    img = cv2.imread(file)
    assert img is not None

    assert check_screen(img, color, find_screen(img, color)) < 30


@mark.parametrize(
    "file, color, wrong_color",
    (
        ("../resources/white 0.jpg", Color.WHITE, Color.BLACK),
        ("../resources/blue 0.jpg", Color.BLUE, Color.WHITE),
        ("../resources/green 0.jpg", Color.GREEN, Color.RED),
        ("../resources/red 0.jpg", Color.RED, Color.BLUE),
    ),
)
def test_check_screen_fail(file: str, color: Color, wrong_color: Color):
    img = cv2.imread(file)
    assert img is not None

    assert check_screen(img, wrong_color, find_screen(img, color)) > 40
