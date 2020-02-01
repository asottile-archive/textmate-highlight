import argparse
import functools
import itertools
import json
import re
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Pattern
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

T = TypeVar('T')

C_256 = '\x1b[38;5;{c}m'
C_TRUE = '\x1b[38;2;{r};{g};{b}m'
C_BG_TRUE = '\x1b[48;2;{r};{g};{b}m'
C_RESET = '\x1b[m'

# yes I know this is wrong, but it's good enough for now
UN_COMMENT = re.compile(r'^\s*//.*$', re.MULTILINE)


class Color(NamedTuple):
    r: int
    g: int
    b: int

    @classmethod
    def parse(cls, s: str) -> 'Color':
        return cls(r=int(s[1:3], 16), g=int(s[3:5], 16), b=int(s[5:7], 16))


def _table_256() -> Dict[Color, int]:
    vals = (0, 95, 135, 175, 215, 255)
    ret = {
        Color(r, g, b): 16 + i
        for i, (r, g, b) in enumerate(itertools.product(vals, vals, vals))
    }
    for i in range(24):
        v = 10 * i + 8
        ret[Color(v, v, v)] = 232 + i
    return ret


TABLE_256 = _table_256()


class Style(NamedTuple):
    foreground: Color
    background: Color
    bold: bool
    italic: bool
    underline: bool


def _select(scope: Tuple[str, ...], rules: Tuple[Tuple[str, T], ...]) -> T:
    assert len(scope) == 1, scope  # this is wrong!
    matches = []
    for selector, color in rules:
        if selector == scope[0]:
            matches.append(color)
    if matches:
        return matches[0]
    else:
        new_scope, _, _ = scope[0].rpartition('.')
        return _select((new_scope,), rules)


class Theme(NamedTuple):
    foreground_rules: Tuple[Tuple[str, Color], ...]
    background_rules: Tuple[Tuple[str, Color], ...]
    bold_rules: Tuple[Tuple[str, bool], ...]
    italic_rules: Tuple[Tuple[str, bool], ...]
    underline_rules: Tuple[Tuple[str, bool], ...]

    @classmethod
    def parse(cls, filename: str) -> 'Theme':
        with open(filename) as f:
            contents = UN_COMMENT.sub('', f.read())
            data = json.loads(contents)

        foregrounds = {'': Color(0xff, 0xff, 0xff)}
        backgrounds = {'': Color(0x00, 0x00, 0x00)}
        bolds = {'': False}
        italics = {'': False}
        underlines = {'': False}

        for k in ('foreground', 'editor.foreground'):
            if k in data['colors']:
                foregrounds[''] = Color.parse(data['colors'][k])
                break

        for k in ('background', 'editor.background'):
            if k in data['colors']:
                backgrounds[''] = Color.parse(data['colors'][k])
                break

        for theme_item in data['tokenColors']:
            if 'scope' not in theme_item:
                scopes = ['']  # some sort of default scope?
            elif isinstance(theme_item['scope'], str):
                scopes = [
                    s.strip() for s in theme_item['scope'].split(',')
                    # some themes have a trailing comma -- do they
                    # intentionally mean to match that? is it a bug? should I
                    # send a patch?
                    if s.strip()
                ]
            else:
                scopes = theme_item['scope']

            for scope in scopes:
                if 'foreground' in theme_item['settings']:
                    color = Color.parse(theme_item['settings']['foreground'])
                    foregrounds[scope] = color
                if 'background' in theme_item['settings']:
                    color = Color.parse(theme_item['settings']['background'])
                if theme_item['settings'].get('fontStyle') == 'bold':
                    bolds[scope] = True
                elif theme_item['settings'].get('fontStyle') == 'italic':
                    italics[scope] = True
                elif theme_item['settings'].get('fontStyle') == 'underline':
                    underlines[scope] = True

        return cls(
            foreground_rules=tuple(foregrounds.items()),
            background_rules=tuple(backgrounds.items()),
            bold_rules=tuple(bolds.items()),
            italic_rules=tuple(italics.items()),
            underline_rules=tuple(underlines.items()),
        )

    # TODO: mypy doesn't properly support putting lru_cache on `_select` above
    # python/mypy#1317

    @functools.lru_cache(maxsize=None)
    def _select_foreground(self, scope: Tuple[str, ...]) -> Color:
        return _select(scope, self.foreground_rules)

    @functools.lru_cache(maxsize=None)
    def _select_background(self, scope: Tuple[str, ...]) -> Color:
        return _select(scope, self.background_rules)

    @functools.lru_cache(maxsize=None)
    def _select_bold(self, scope: Tuple[str, ...]) -> bool:
        return _select(scope, self.bold_rules)

    @functools.lru_cache(maxsize=None)
    def _select_italic(self, scope: Tuple[str, ...]) -> bool:
        return _select(scope, self.italic_rules)

    @functools.lru_cache(maxsize=None)
    def _select_underline(self, scope: Tuple[str, ...]) -> bool:
        return _select(scope, self.underline_rules)

    @functools.lru_cache(maxsize=None)
    def select(self, scope: Tuple[str, ...]) -> Style:
        return Style(
            foreground=self._select_foreground(scope),
            background=self._select_background(scope),
            bold=self._select_bold(scope),
            italic=self._select_italic(scope),
            underline=self._select_underline(scope),
        )


class _Rule(Protocol):
    """hax for recursive types python/mypy#731"""
    @property
    def name(self) -> Optional[str]: ...
    @property
    def match(self) -> Optional[Pattern[str]]: ...
    @property
    def begin(self) -> Optional[Pattern[str]]: ...
    @property
    def end(self) -> Optional[str]: ...
    @property
    def content_name(self) -> Optional[str]: ...
    @property
    def captures(self) -> Dict[int, str]: ...
    @property
    def begin_captures(self) -> Dict[int, str]: ...
    @property
    def end_captures(self) -> Dict[int, str]: ...
    @property
    def include(self) -> Optional[str]: ...
    @property
    def patterns(self) -> 'Tuple[_Rule, ...]': ...


class Rule(NamedTuple):
    name: Optional[str]
    match: Optional[Pattern[str]]
    begin: Optional[Pattern[str]]
    end: Optional[str]
    content_name: Optional[str]
    captures: Dict[int, str]
    begin_captures: Dict[int, str]
    end_captures: Dict[int, str]
    include: Optional[str]
    patterns: Tuple[_Rule, ...]

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> _Rule:
        if 'name' in dct:
            name = dct['name']
        else:
            name = None

        if 'match' in dct:
            match: Optional[Pattern[str]] = re.compile(dct['match'])
        else:
            match = None

        if 'begin' in dct:
            begin: Optional[Pattern[str]] = re.compile(dct['begin'])
        else:
            begin = None

        # end can have back references so we lazily compile this
        if 'end' in dct:
            end = dct['end']
        else:
            end = None

        if 'contentName' in dct:
            content_name = dct['contentName']
        else:
            content_name = None

        if 'captures' in dct:
            captures = {int(k): v['name'] for k, v in dct['captures'].items()}
        else:
            captures = {}

        if 'beginCaptures' in dct:
            begin_captures = {
                int(k): v['name'] for k, v in dct['beginCaptures'].items()
            }
        else:
            begin_captures = {}

        if 'endCaptures' in dct:
            end_captures = {
                int(k): v['name'] for k, v in dct['endCaptures'].items()
            }
        else:
            end_captures = {}

        if 'include' in dct:
            include = dct['include']
        else:
            include = None

        if 'patterns' in dct:
            patterns = tuple(Rule.from_dct(d) for d in dct['patterns'])
        else:
            patterns = ()

        return cls(
            name=name,
            match=match,
            begin=begin,
            end=end,
            content_name=content_name,
            captures=captures,
            begin_captures=begin_captures,
            end_captures=end_captures,
            include=include,
            patterns=patterns,
        )


class Grammar(NamedTuple):
    scope_name: str
    first_line_match: Optional[str]
    file_types: Tuple[str, ...]
    patterns: Tuple[_Rule, ...]
    repository: Dict[str, _Rule]

    @classmethod
    def parse(cls, filename: str) -> 'Grammar':
        with open(filename) as f:
            contents = UN_COMMENT.sub('', f.read())
            data = json.loads(contents)

        scope_name = data['scopeName']
        if 'firstLineMatch' in data:
            first_line_match = data['firstLineMatch']
        else:
            first_line_match = None
        if 'fileTypes' in data:
            file_types = tuple(data['fileTypes'])
        else:
            file_types = ()
        patterns = tuple(Rule.from_dct(dct) for dct in data['patterns'])
        if 'repository' in data:
            repository = {
                k: Rule.from_dct(dct) for k, dct in data['repository'].items()
            }
        else:
            repository = {}
        return cls(
            scope_name=scope_name,
            first_line_match=first_line_match,
            file_types=file_types,
            patterns=patterns,
            repository=repository,
        )


def print_styled(s: str, style: Style) -> None:
    color_s = ''
    undo_s = ''
    color_s += C_TRUE.format(**style.foreground._asdict())
    color_s += C_BG_TRUE.format(**style.background._asdict())
    undo_s += '\x1b[39m'
    if style.bold:
        color_s += '\x1b[1m'
        undo_s += '\x1b[22m'
    if style.italic:
        color_s += '\x1b[3m'
        undo_s += '\x1b[23m'
    if style.underline:
        color_s += '\x1b[4m'
        undo_s += '\x1b[24m'
    print(f'{color_s}{s}{undo_s}', end='')


def _highlight(theme_filename: str, syntax_filename: str, file: str) -> int:

    theme = Theme.parse(theme_filename)
    grammar = Grammar.parse(syntax_filename)

    with open(file) as f:
        lines = list(f)

    print(C_BG_TRUE.format(**theme.select(('',)).background._asdict()))
    lineno = 0
    pos = 0
    while lineno < len(lines):
        line = lines[lineno]
        for rule in grammar.patterns:
            if rule.match is not None:
                match = rule.match.match(line, pos)
                if match is not None:
                    # XXX: this should include the file type and full path
                    style = theme.select((rule.name,))
                    print_styled(match[0], style)
                    pos = match.end()  # + 1 ?
                    if pos >= len(line):
                        lineno += 1
                        pos = 0
                    break
            # start / end based one -- how to do this?
            elif rule.begin is not None:
                assert rule.end is not None
                if rule.begin.match(line):
                    raise NotImplementedError('begin/end tokens')
                    break
            else:
                raise AssertionError('unreachable!')
        else:
            print_styled(line[pos], theme.select(('',)))
            pos += 1
            if pos >= len(line):
                lineno += 1
                pos = 0

    print('\x1b[m', end='')
    return 0


def _theme(theme_filename: str) -> int:
    theme = Theme.parse(theme_filename)

    print(C_BG_TRUE.format(**theme.select(('',)).background._asdict()))
    print_styled('(DEFAULT)\n', theme.select(('',)))
    all_styles = {''}
    all_styles.update((k for k, v in theme.foreground_rules))
    all_styles.update((k for k, v in theme.background_rules))
    all_styles.update((k for k, v in theme.bold_rules))
    all_styles.update((k for k, v in theme.italic_rules))
    all_styles.update((k for k, v in theme.underline_rules))
    all_styles.discard('')
    for k in sorted(all_styles):
        print_styled(f'{k}\n', theme.select((k,)))
    print('\x1b[m', end='')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    highlight_parser = subparsers.add_parser('highlight')
    highlight_parser.add_argument('theme')
    highlight_parser.add_argument('syntax')
    highlight_parser.add_argument('file')

    theme_parser = subparsers.add_parser('theme')
    theme_parser.add_argument('theme')

    args = parser.parse_args()

    if args.command == 'highlight':
        return _highlight(args.theme, args.syntax, args.file)
    elif args.command == 'theme':
        return _theme(args.theme)
    else:
        raise NotImplementedError(args.command)


if __name__ == '__main__':
    exit(main())