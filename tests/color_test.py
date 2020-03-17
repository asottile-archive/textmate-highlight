import pytest

from highlight_demo.color import Color


@pytest.mark.parametrize(
    ('s', 'expected'),
    (
        ('#1e77d3', Color(0x1e, 0x77, 0xd3)),
        ('white', Color(0xff, 0xff, 0xff)),
        ('black', Color(0x00, 0x00, 0x00)),
    ),
)
def test_color_parse(s, expected):
    assert Color.parse(s) == expected
