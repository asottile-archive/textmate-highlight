import pytest

from highlight_demo import _select
from highlight_demo import Selector


RULES = (
    (Selector.parse(''), 'trivial selector'),
    (Selector.parse('foo.bar'), 'foo bar selector'),
    (Selector.parse('foo'), 'foo selector'),
    (Selector.parse('foo bar'), 'foo bar descendant selector'),
    (Selector.parse('foo > bar'), 'foo bar child selector'),
    (Selector.parse('foo - bar'), 'foo bar negation selector'),
    (Selector.parse('*wc*'), 'wc wildcard selector'),
)


@pytest.mark.parametrize(
    ('scope', 'expected'),
    (
        pytest.param(('',), 'trivial selector', id='trivial'),
        pytest.param(('unknown',), 'trivial selector', id='unknown'),
        pytest.param(('foo.bar',), 'foo bar selector', id='exact match'),
        pytest.param(('foo.baz',), 'foo selector', id='prefix match'),
        pytest.param(
            ('src.diff', 'foo.bar'), 'foo bar selector',
            id='nested scope',
        ),
        pytest.param(
            ('foo.bar', 'unrelated'), 'foo bar selector',
            id='nested scope not last one',
        ),
    ),
)
def test_select(scope, expected):
    assert _select(scope, RULES) == expected
