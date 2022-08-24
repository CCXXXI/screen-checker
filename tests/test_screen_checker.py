from itertools import chain, product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytest import mark, raises

from screen_checker import Color, find_screen, check_screen, get_lengths, ocr_ssd

PASS_LIMIT = FAIL_LIMIT = 25

photos = {c.name: set(c.iterdir()) for c in Path("../resources/").iterdir()}


@mark.parametrize("file", chain.from_iterable(photos.values()))
def test_imread(file: Path):
    assert cv2.imread(file.as_posix()) is not None


@mark.parametrize(
    "file, color",
    (
        *product(photos["white"], [Color.WHITE, Color.BLUE, Color.GREEN, Color.RED]),
        *product(photos["blue"], [Color.BLUE]),
        *product(photos["green"], [Color.GREEN]),
        *product(photos["red"], [Color.RED]),
    ),
)
def test_find_screen(file: Path, color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    corners = find_screen(img, color)
    assert corners.shape == (4, 2)
    assert np.all(corners >= 0)


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
        *product(photos["white"], [Color.WHITE, Color.BLUE, Color.GREEN, Color.RED]),
        *product(photos["blue"], [Color.BLUE]),
        *product(photos["green"], [Color.GREEN]),
        *product(photos["red"], [Color.RED]),
    ),
)
def test_get_lengths(file: Path, color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    corners = find_screen(img, color)
    lengths = get_lengths(corners)

    assert abs(lengths[0] / lengths[2] - 1) < 0.1
    assert abs(lengths[1] / lengths[3] - 1) < 0.1


@mark.parametrize(
    "file, color",
    (
        *product(photos["white"], [Color.WHITE]),
        *product(photos["blue"], [Color.BLUE]),
        *product(photos["green"], [Color.GREEN]),
        *product(photos["red"], [Color.RED]),
    ),
)
def test_check_screen_pass(file: Path, color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    assert check_screen(img, color, find_screen(img, color)) < PASS_LIMIT


@mark.parametrize(
    "black, white",
    product(photos["black"], [Path("../resources/white/0.png")]),
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
        *product(
            photos["white"],
            [Color.WHITE],
            [Color.BLUE, Color.GREEN, Color.RED, Color.BLACK],
        ),
        *product(
            photos["blue"],
            [Color.BLUE],
            [Color.WHITE, Color.GREEN, Color.RED, Color.BLACK],
        ),
        *product(
            photos["green"],
            [Color.GREEN],
            [Color.WHITE, Color.BLUE, Color.RED, Color.BLACK],
        ),
        *product(
            photos["red"],
            [Color.RED],
            [Color.WHITE, Color.BLUE, Color.GREEN, Color.BLACK],
        ),
    ),
)
def test_check_screen_fail(file: Path, color: Color, wrong_color: Color):
    img = cv2.imread(file.as_posix())
    assert img is not None

    assert check_screen(img, wrong_color, find_screen(img, color)) > FAIL_LIMIT


@mark.parametrize(
    "black, white, wrong_color",
    product(
        photos["black"],
        [Path("../resources/white/0.png")],
        [Color.BLUE, Color.GREEN, Color.RED, Color.WHITE],
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


@mark.parametrize(
    "file, text",
    product(photos["ocr"], ["128"]),
)
def test_ocr_ssd(file: Path, text: str):
    img = cv2.imread(file.as_posix())
    assert img is not None

    assert ocr_ssd(img) == text
    assert ocr_ssd(img[::-1, :, :]) != text
    assert ocr_ssd(img[:, ::-1, :]) != text
    assert ocr_ssd(img[::-1, ::-1, :]) != text


def test_debug():
    import screen_checker

    screen_checker.debug = True

    img = cv2.imread("../resources/white/0.png")
    assert img is not None
    assert check_screen(img, Color.WHITE, find_screen(img, Color.WHITE)) < PASS_LIMIT
    plt.close("all")

    assert ocr_ssd(img) is not None
    plt.close("all")

    img = cv2.imread("../resources/green/0.png")
    assert img is not None
    assert check_screen(img, Color.GREEN, find_screen(img, Color.GREEN)) < PASS_LIMIT
    plt.close("all")

    screen_checker.debug = False
