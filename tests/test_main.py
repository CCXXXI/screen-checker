from src.main import add
from pytest import mark


@mark.parametrize(
    "a, b, expected",
    (
        (1, 2, 3),
        (3.14, 2.71, 5.85),
        ("a", "b", "ab"),
        ([1, 2], [True, False], [1, 2, True, False]),
    ),
)
def test_add(a, b, expected):
    assert add(a, b) == expected
