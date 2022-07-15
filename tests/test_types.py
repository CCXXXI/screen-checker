from src.types import Quad, Point
from math import dist, sqrt

quad = Quad(
    top_left=Point(row=0, col=0),
    top_right=Point(row=0, col=2),
    bottom_left=Point(row=1, col=0),
    bottom_right=Point(row=1, col=3),
)


def test_dist():
    tl, tr, bl, br = quad
    assert dist(tl, tr) == 2
    assert dist(tl, bl) == 1
    assert dist(tl, br) == sqrt(10)
    assert dist(tr, bl) == sqrt(5)
