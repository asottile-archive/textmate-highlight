import pytest

from highlight_demo import _remove_comments
from highlight_demo import _select
from highlight_demo import _tokenize_re
from highlight_demo import re_compile
from highlight_demo import Selector


RULES = (
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
    assert _select(scope, RULES, 'trivial selector') == expected


def test_re_compile_can_compile_a_regex():
    pattern = re_compile('hello wo+rld')
    assert pattern.match('hello woooooooooooooorld')


def test_re_compile_possessive_quantifier_star():
    pattern = re_compile('".*+"')
    assert not pattern.match('"abc"')
    assert not pattern.match('"abc"x')


def test_re_compile_possessive_quantifier_plus():
    pattern = re_compile('".++"')
    assert not pattern.match('"abc"')
    assert not pattern.match('"abc"x')


def test_re_compile_multiple_possessive_quantifiers():
    pattern = re_compile('"a*+" "b*+"')
    assert pattern.match('"aaaaaa" "bbbbbbb"')


def test_re_compile_not_a_possessive_quantifier():
    pattern = re_compile(r'\*+.q')
    assert pattern.match('*****q')


def test_re_compile_character_class():
    pattern = re_compile(r'"\s*+"')
    assert pattern.groupindex
    assert pattern.match('"    "')


def test_re_compile_character_set():
    pattern = re_compile('"[^"]++"')
    assert pattern.groupindex
    assert pattern.match('"hello world"')


def test_re_compile_group():
    pattern = re_compile('"(ab)++"')
    assert pattern.groupindex
    assert pattern.match('"ababab"')


def test_re_compile_unicode_escape():
    pattern = re_compile(r'"\u2603++"')
    assert pattern.groupindex
    assert pattern.match('"☃☃☃☃"')


def test_re_compile_possessive_in_comment():
    pattern = re_compile('(?# this is awesome, asottile++)hello')
    assert not pattern.groupindex
    assert pattern.match('hello')


def test_re_compile_hex_escape():
    pattern = re_compile(r'\h hello')
    assert pattern.match('a hello')
    assert pattern.match('A hello')
    assert pattern.match('9 hello')
    assert not pattern.match('q hello')


def test_re_compile_hex_negation_escape():
    pattern = re_compile(r'\H hello')
    assert pattern.match('q hello')
    assert not pattern.match('a hello')


@pytest.mark.parametrize(
    ('s', 'expected'),
    (
        ('', ''),
        ('foo', 'foo'),
        ('(?#foo)', ''),
        (r'(?#foo \) bar)baz', 'baz'),
        ('(?#foo\nbar)baz', 'baz'),
        ('hello(?#wat)world', 'helloworld'),
    ),
)
def test_remove_comments(s, expected):
    assert _remove_comments(s) == expected


@pytest.mark.parametrize(
    ('s', 'expected'),
    (
        ('', {}),
        ('foo', {}),
        ('(foo)', {4: 0}),
        ('[foo]', {4: 0}),
        ('[fo)]', {4: 0}),
        ('(foo[bar])', {8: 4, 9: 0}),
        (r'\(foo\)', {1: 0, 6: 5}),
        (r'\\(foo)', {1: 0, 6: 2}),
        ('[]hi]', {4: 0}),
        (r'\s', {1: 0}),
        (r'\xa0aaa', {3: 0}),
        (r'\xA0aaa', {3: 0}),
        (r'\u2603aaa', {5: 0}),
        (r'\u259Aaaa', {5: 0}),
        (r'\U0001f643aaa', {9: 0}),
        (r'\U0001F643aaa', {9: 0}),
        (r'\0zzz', {1: 0}),
        (r'\00zzz', {2: 0}),
        (r'\01zzz', {2: 0}),
        (r'\000zzz', {3: 0}),
        (r'\001zzz', {3: 0}),
        (r'\101zzz', {3: 0}),
        (r'\1aaa', {1: 0}),
        (r'\99aaa', {2: 0}),
        (r'\199', {2: 0}),
    ),
)
def test_tokenize_re(s, expected):
    assert _tokenize_re(s) == expected
