from itertools import product
from pathlib import Path

import cv2
import numpy as np
from pytest import mark, raises

from screen_checker import Color, find_screen, check_screen

PASS_LIMIT = FAIL_LIMIT = 30

photos = {
    c: set(Path(f"../resources/{c}/").glob("*"))
    for c in ("blue", "green", "red", "white", "black")
}


@mark.parametrize(
    "file, color",
    (
        *product(photos["white"], set(Color) - {Color.BLACK}),
        *product(photos["blue"], {Color.BLUE}),
        *product(photos["green"], {Color.GREEN}),
        *product(photos["red"], {Color.RED}),
    ),
)
def test_find_screen(file: Path, color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    corners = find_screen(img, color)
    assert corners.shape == (4, 2)
    assert np.all(corners > 0)


@mark.parametrize(
    "file, color",
    product(photos["black"], Color),
)
def test_find_screen_error(file: Path, color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    with raises(ValueError):
        find_screen(img, color)


@mark.parametrize(
    "file, color",
    (
        *product(photos["white"], {Color.WHITE}),
        *product(photos["blue"], {Color.BLUE}),
        *product(photos["green"], {Color.GREEN}),
        *product(photos["red"], {Color.RED}),
    ),
)
def test_check_screen_pass(file: Path, color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    assert check_screen(img, color, find_screen(img, color)) < PASS_LIMIT


@mark.parametrize(
    "black, white",
    product(photos["black"], {Path("../resources/white/0.png")}),
)
def test_check_screen_pass_black(black: Path, white: Path):
    img_white = cv2.imread(white.as_posix())
    img_black = cv2.imread(black.as_posix())
    assert img_white is not None
    assert img_black is not None

    assert (
        check_screen(img_black, Color.BLACK, find_screen(img_white, Color.WHITE))
        < PASS_LIMIT
    )


@mark.parametrize(
    "file, color, wrong_color",
    (
        *product(photos["white"], {Color.WHITE}, set(Color) - {Color.WHITE}),
        *product(photos["blue"], {Color.BLUE}, set(Color) - {Color.BLUE}),
        *product(photos["green"], {Color.GREEN}, set(Color) - {Color.GREEN}),
        *product(photos["red"], {Color.RED}, set(Color) - {Color.RED}),
    ),
)
def test_check_screen_fail(file: Path, color: Color, wrong_color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    assert check_screen(img, wrong_color, find_screen(img, color)) > FAIL_LIMIT


@mark.parametrize(
    "black, white, wrong_color",
    product(
        photos["black"], {Path("../resources/white/0.png")}, set(Color) - {Color.BLACK}
    ),
)
def test_check_screen_fail_black(black: Path, white: Path, wrong_color: Color):
    img_white = cv2.imread(white.as_posix())
    img_black = cv2.imread(black.as_posix())
    assert img_white is not None
    assert img_black is not None

    assert (
        check_screen(img_black, wrong_color, find_screen(img_white, Color.WHITE))
        > FAIL_LIMIT
    )


def test_debug():
    import screen_checker

    screen_checker.debug = True

    img = cv2.imread("../resources/white/0.png")
    assert img is not None
    assert check_screen(img, Color.WHITE, find_screen(img, Color.WHITE)) < PASS_LIMIT

    screen_checker.debug = False
