import pytest

from highlight_demo.highlight import Color
from highlight_demo.highlight import Entry
from highlight_demo.highlight import Grammar
from highlight_demo.highlight import Grammars
from highlight_demo.highlight import highlight_line
from highlight_demo.highlight import Region
from highlight_demo.highlight import State
from highlight_demo.highlight import Theme

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
    assert unhex(ret.fg) == unhex(Color.parse(expected))


def _compiler_state(grammar_dct):
    grammar = Grammar.from_data(grammar_dct)
    compiler = Grammars([grammar]).compiler_for_scope('test')
    state = State.root(Entry(compiler.root.name, compiler.root))
    return compiler, state


def test_backslash_a():
    grammar = {
        'scopeName': 'test',
        'patterns': [{'name': 'aaa', 'match': r'\Aa+'}],
    }
    compiler, state = _compiler_state(grammar)

    state, (region_0,) = highlight_line(compiler, state, 'aaa', True)
    state, (region_1,) = highlight_line(compiler, state, 'aaa', False)

    # \A should only match at the beginning of the file
    assert region_0 == Region(0, 3, ('test', 'aaa'))
    assert region_1 == Region(0, 3, ('test',))


BEGIN_END_NO_NL = {
    'scopeName': 'test',
    'patterns': [{
        'begin': 'x',
        'end': 'x',
        'patterns': [
            {'match': r'\Ga', 'name': 'ga'},
            {'match': 'a', 'name': 'noga'},
        ],
    }],
}


def test_backslash_g_inline():
    compiler, state = _compiler_state(BEGIN_END_NO_NL)

    _, regions = highlight_line(compiler, state, 'xaax', True)
    assert regions == (
        Region(0, 1, ('test',)),
        Region(1, 2, ('test', 'ga')),
        Region(2, 3, ('test', 'noga')),
        Region(3, 4, ('test',)),
    )


def test_backslash_g_next_line():
    compiler, state = _compiler_state(BEGIN_END_NO_NL)

    state, regions1 = highlight_line(compiler, state, 'x\n', True)
    state, regions2 = highlight_line(compiler, state, 'aax\n', False)

    assert regions1 == (
        Region(0, 1, ('test',)),
        Region(1, 2, ('test',)),
    )
    assert regions2 == (
        Region(0, 1, ('test', 'noga')),
        Region(1, 2, ('test', 'noga')),
        Region(2, 3, ('test',)),
        Region(3, 4, ('test',)),
    )


BEGIN_END_NL = {
    'scopeName': 'test',
    'patterns': [{
        'begin': r'x$\n?',
        'end': 'x',
        'patterns': [
            {'match': r'\Ga', 'name': 'ga'},
            {'match': 'a', 'name': 'noga'},
        ],
    }],
}


def test_backslash_g_captures_nl():
    compiler, state = _compiler_state(BEGIN_END_NL)

    state, regions1 = highlight_line(compiler, state, 'x\n', True)
    state, regions2 = highlight_line(compiler, state, 'aax\n', False)

    assert regions1 == (
        Region(0, 2, ('test',)),
    )
    assert regions2 == (
        Region(0, 1, ('test', 'ga')),
        Region(1, 2, ('test', 'noga')),
        Region(2, 3, ('test',)),
        Region(3, 4, ('test',)),
    )


def test_backslash_g_captures_nl_next_line():
    compiler, state = _compiler_state(BEGIN_END_NL)

    state, regions1 = highlight_line(compiler, state, 'x\n', True)
    state, regions2 = highlight_line(compiler, state, 'aa\n', False)
    state, regions3 = highlight_line(compiler, state, 'aax\n', False)

    assert regions1 == (
        Region(0, 2, ('test',)),
    )
    assert regions2 == (
        Region(0, 1, ('test', 'ga')),
        Region(1, 2, ('test', 'noga')),
        Region(2, 3, ('test',)),
    )
    assert regions3 == (
        Region(0, 1, ('test', 'ga')),
        Region(1, 2, ('test', 'noga')),
        Region(2, 3, ('test',)),
        Region(3, 4, ('test',)),
    )


def test_while_no_nl():
    compiler, state = _compiler_state({
        'scopeName': 'test',
        'patterns': [{
            'begin': '> ',
            'while': '> ',
            'patterns': [
                {'match': r'\Ga', 'name': 'ga'},
                {'match': 'a', 'name': 'noga'},
            ],
        }],
    })

    state, regions1 = highlight_line(compiler, state, '> aa\n', True)
    state, regions2 = highlight_line(compiler, state, '> aa\n', False)

    assert regions1 == regions2 == (
        Region(0, 2, ('test',)),
        Region(2, 3, ('test', 'ga')),
        Region(3, 4, ('test', 'noga')),
        Region(4, 5, ('test',)),
    )
