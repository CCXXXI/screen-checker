from typing import NamedTuple
from numpy import typing as npt

Image = npt.NDArray


class Point(NamedTuple):
    row: int
    col: int


class Quad(NamedTuple):
    top_left: Point
    top_right: Point
    bottom_left: Point
    bottom_right: Point
