from teststand_helper import find_screen, get_lengths, get_size


def test_find_screen():
    corners = find_screen("../resources/white/1.png")

    assert len(corners) == 8


def test_find_screen_fail():
    corners = find_screen("../resources/white/0.png")

    assert len(corners) == 0


def test_get_lengths():
    corners = find_screen("../resources/white/1.png")
    lengths = get_lengths(corners)

    assert len(lengths) == 4


def test_get_size():
    corners = find_screen("../resources/white/1.png")
    size = get_size(corners)

    assert size > 0
