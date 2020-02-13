import argparse
import functools
import itertools
import json
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Match
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar

import onigurumacffi

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

compile_regset = functools.lru_cache()(onigurumacffi.compile_regset)

T = TypeVar('T')
Scope = Tuple[str, ...]

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
    fg: Color
    bg: Color
    b: bool
    i: bool
    u: bool


class Selector(NamedTuple):
    # TODO: parts: Tuple[str, ...]
    s: str

    @classmethod
    def parse(cls, s: str) -> 'Selector':
        return cls(s)

    def matches(self, scope: Scope) -> Tuple[bool, int]:
        s = scope[-1]
        while s:
            if s == self.s:
                return (True, s.count('.') + bool(s))
            s, _, _ = s.rpartition('.')
        return (False, -1)


DEFAULT_SELECTOR = Selector.parse('')


def _select(
        scope: Scope,
        rules: Tuple[Tuple[Selector, T], ...],
        default: T,
) -> T:
    for scope_len in range(len(scope), 0, -1):
        sub_scope = scope[:scope_len]
        matches = []
        for selector, t in rules:
            is_matched, priority = selector.matches(sub_scope)
            if is_matched:
                matches.append((priority, t))
        if matches:  # TODO: and len(matches) == 1
            _, ret = max(matches)
            return ret

    return default


class Theme(NamedTuple):
    fg_default: Color
    bg_default: Color
    b_default: bool
    i_default: bool
    u_default: bool
    fg_rules: Tuple[Tuple[Selector, Color], ...]
    bg_rules: Tuple[Tuple[Selector, Color], ...]
    b_rules: Tuple[Tuple[Selector, bool], ...]
    i_rules: Tuple[Tuple[Selector, bool], ...]
    u_rules: Tuple[Tuple[Selector, bool], ...]

    @classmethod
    def parse(cls, filename: str) -> 'Theme':
        with open(filename) as f:
            contents = UN_COMMENT.sub('', f.read())
            data = json.loads(contents)

        fg_d = {DEFAULT_SELECTOR: Color(0xff, 0xff, 0xff)}
        bg_d = {DEFAULT_SELECTOR: Color(0x00, 0x00, 0x00)}
        b_d = {DEFAULT_SELECTOR: False}
        i_d = {DEFAULT_SELECTOR: False}
        u_d = {DEFAULT_SELECTOR: False}

        for k in ('foreground', 'editor.foreground'):
            if k in data['colors']:
                fg_d[DEFAULT_SELECTOR] = Color.parse(data['colors'][k])
                break

        for k in ('background', 'editor.background'):
            if k in data['colors']:
                bg_d[DEFAULT_SELECTOR] = Color.parse(data['colors'][k])
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
                selector = Selector.parse(scope)
                if 'foreground' in theme_item['settings']:
                    color = Color.parse(theme_item['settings']['foreground'])
                    fg_d[selector] = color
                if 'background' in theme_item['settings']:
                    color = Color.parse(theme_item['settings']['background'])
                    bg_d[selector] = color
                if theme_item['settings'].get('fontStyle') == 'bold':
                    b_d[selector] = True
                elif theme_item['settings'].get('fontStyle') == 'italic':
                    i_d[selector] = True
                elif theme_item['settings'].get('fontStyle') == 'underline':
                    u_d[selector] = True

        return cls(
            fg_default=fg_d.pop(DEFAULT_SELECTOR),
            bg_default=bg_d.pop(DEFAULT_SELECTOR),
            b_default=b_d.pop(DEFAULT_SELECTOR),
            i_default=i_d.pop(DEFAULT_SELECTOR),
            u_default=u_d.pop(DEFAULT_SELECTOR),
            fg_rules=tuple(fg_d.items()),
            bg_rules=tuple(bg_d.items()),
            b_rules=tuple(b_d.items()),
            i_rules=tuple(i_d.items()),
            u_rules=tuple(u_d.items()),
        )

    @functools.lru_cache(maxsize=None)
    def select(self, scope: Scope) -> Style:
        return Style(
            fg=_select(scope, self.fg_rules, self.fg_default),
            bg=_select(scope, self.bg_rules, self.bg_default),
            b=_select(scope, self.b_rules, self.b_default),
            i=_select(scope, self.i_rules, self.i_default),
            u=_select(scope, self.u_rules, self.u_default),
        )


class _Rule(Protocol):
    """hax for recursive types python/mypy#731"""
    @property
    def name(self) -> Optional[str]: ...
    @property
    def match(self) -> Optional[str]: ...
    @property
    def begin(self) -> Optional[str]: ...
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
    match: Optional[str]
    begin: Optional[str]
    end: Optional[str]
    content_name: Optional[str]
    captures: Dict[int, str]
    begin_captures: Dict[int, str]
    end_captures: Dict[int, str]
    include: Optional[str]
    patterns: Tuple[_Rule, ...]

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> _Rule:
        name = dct.get('name')
        match = dct.get('match')
        begin = dct.get('begin')
        end = dct.get('end')
        content_name = dct.get('contentName')

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

        include = dct.get('include')

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
    color_s += C_TRUE.format(**style.fg._asdict())
    color_s += C_BG_TRUE.format(**style.bg._asdict())
    undo_s += '\x1b[39m'
    if style.b:
        color_s += '\x1b[1m'
        undo_s += '\x1b[22m'
    if style.i:
        color_s += '\x1b[3m'
        undo_s += '\x1b[23m'
    if style.u:
        color_s += '\x1b[4m'
        undo_s += '\x1b[24m'
    print(f'{color_s}{s}{undo_s}', end='')


StyleCB = Callable[[Match[str]], Style]


class Entry(NamedTuple):
    regset: onigurumacffi._RegSet
    callbacks: Tuple[StyleCB, ...]
    scopes: Tuple[str, ...]


def _highlight(theme_filename: str, syntax_filename: str, file: str) -> int:
    theme = Theme.parse(theme_filename)
    grammar = Grammar.parse(syntax_filename)

    with open(file) as f:
        lines = list(f)

    print(C_BG_TRUE.format(**theme.bg_default._asdict()))
    lineno = 0
    pos = 0
    stack: List[Entry] = []

    def _entry(
            patterns: Tuple[_Rule, ...],
            end: str,
            scope: Tuple[str, ...],
    ) -> Entry:
        def _end_cb(match: Match[str]) -> Style:
            stack.pop()
            return theme.select(entry.scopes)

        def _match_cb(match: Match[str], *, rule: _Rule) -> Style:
            if rule.name is None:
                return theme.select(entry.scopes)
            else:
                return theme.select((*entry.scopes, rule.name))

        def _begin_cb(match: Match[str], *, rule: _Rule) -> Style:
            assert rule.end is not None
            if rule.name is not None:
                next_scopes = (*entry.scopes, rule.name)
            else:
                next_scopes = entry.scopes

            end = match.expand(rule.end)
            stack.append(_entry(rule.patterns, end, next_scopes))

            return theme.select(next_scopes)

        regs = [end]
        cbs = [_end_cb]

        rules = list(reversed(patterns))
        while rules:
            rule = rules.pop()

            # XXX: can a rule have an include also?
            if rule.include is not None:
                assert rule.match is None
                assert rule.begin is None
                rule = grammar.repository[rule.include[1:]]

            if rule.match is None and rule.begin is None and rule.patterns:
                rules.extend(reversed(rule.patterns))
            elif rule.match is not None:
                regs.append(rule.match)
                cbs.append(functools.partial(_match_cb, rule=rule))
            elif rule.begin is not None:
                regs.append(rule.begin)
                cbs.append(functools.partial(_begin_cb, rule=rule))
            else:
                raise AssertionError(f'unreachable {rule}')

        return Entry(compile_regset(*regs), tuple(cbs), scope)

    stack.append(_entry(grammar.patterns, ' ^', (grammar.scope_name,)))
    while lineno < len(lines):
        line = lines[lineno]
        entry = stack[-1]

        idx, match = entry.regset.search(line, pos)
        if match is not None:
            print_styled(line[pos:match.start()], theme.select(entry.scopes))

            style = entry.callbacks[idx](match)
            print_styled(match[0], style)

            pos = match.end()
            if pos >= len(line):
                lineno += 1
                pos = 0
        else:
            print_styled(line[pos:], theme.select(entry.scopes))
            lineno += 1
            pos = 0

    print('\x1b[m', end='')
    return 0


def _theme(theme_filename: str) -> int:
    theme = Theme.parse(theme_filename)

    fg_d = dict(theme.fg_rules)
    bg_d = dict(theme.bg_rules)
    b_d = dict(theme.b_rules)
    i_d = dict(theme.i_rules)
    u_d = dict(theme.u_rules)

    print(C_BG_TRUE.format(**theme.bg_default._asdict()))
    rules = {DEFAULT_SELECTOR}.union(fg_d, bg_d, b_d, i_d, u_d)
    for k in sorted(rules):
        style = Style(
            fg=fg_d.get(k, theme.fg_default),
            bg=bg_d.get(k, theme.bg_default),
            b=b_d.get(k, theme.b_default),
            i=i_d.get(k, theme.i_default),
            u=u_d.get(k, theme.u_default),
        )
        print_styled(f'{k}\n', style)
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
