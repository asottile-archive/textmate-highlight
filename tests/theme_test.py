import pytest

from highlight_demo.theme import Theme

THEME = Theme.from_dct({
    'colors': {'foreground': '#100000', 'background': '#aaaaaa'},
    'tokenColors': [
        {'scope': 'foo.bar', 'settings': {'foreground': '#200000'}},
        {'scope': 'foo', 'settings': {'foreground': '#300000'}},
        {'scope': 'parent foo.bar', 'settings': {'foreground': '#400000'}},
    ],
})


def unhex(color):
    return f'#{hex(color.r << 16 | color.g << 8 | color.b)[2:]}'


@pytest.mark.parametrize(
    ('scope', 'expected'),
    (
        pytest.param(('',), '#100000', id='trivial'),
        pytest.param(('unknown',), '#100000', id='unknown'),
        pytest.param(('foo.bar',), '#200000', id='exact match'),
        pytest.param(('foo.baz',), '#300000', id='prefix match'),
        pytest.param(('src.diff', 'foo.bar'), '#200000', id='nested scope'),
        pytest.param(
            ('foo.bar', 'unrelated'), '#200000',
            id='nested scope not last one',
        ),
    ),
)
def test_select(scope, expected):
    ret = THEME.select(scope)
    assert unhex(ret.fg) == expected
