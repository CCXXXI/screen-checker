"""Call functions in screen_checker with only numbers, booleans, strings, lists or tuples."""

import cv2
import numpy as np

import screen_checker as sc


def find_screen(photo: str, color: str = "WHITE") -> tuple[int]:
    """
    Find the screen in the photo.

    :param photo: A photo of the screen.
    :param color: The color of the screen. Cannot be black.
    :return: Four (x, y) points which are the four corners of the screen.
    """
    return tuple(sc.find_screen(cv2.imread(photo), sc.Color[color]).flat)


def get_lengths(corners: tuple[int]) -> tuple[float]:
    """
    Get the lengths of the four sides of the screen.

    :param corners: The result of find_screen.
    :return: A list of four float values.
    """
    return sc.get_lengths(np.array(corners).reshape(4, 2))


def get_size(corner: tuple[int]) -> float:
    """
    Get the size of the screen.

    :param corner: The result of find_screen.
    :return: The size of the screen.
    """
    return sc.get_size(np.array(corner).reshape(4, 2))
