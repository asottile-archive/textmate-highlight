import argparse
import curses
import functools
import itertools
import json
import os.path
import plistlib
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Generator
from typing import Generic
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
TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
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

    def as_curses(self) -> Tuple[int, int, int]:
        return (
            int(1000 * self.r / 255),
            int(1000 * self.g / 255),
            int(1000 * self.b / 255),
        )


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
        if self.s == s or s.startswith(f'{self.s}.'):
            return (True, self.s.count('.'))
        else:
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
        if matches:
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


Captures = Tuple[Tuple[int, '_Rule'], ...]


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
    def captures(self) -> Captures: ...
    @property
    def begin_captures(self) -> Captures: ...
    @property
    def end_captures(self) -> Captures: ...
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
    captures: Captures
    begin_captures: Captures
    end_captures: Captures
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
            captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['captures'].items()
            )
        else:
            captures = ()

        if 'beginCaptures' in dct:
            begin_captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['beginCaptures'].items()
            )
        else:
            begin_captures = ()

        if 'endCaptures' in dct:
            end_captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['endCaptures'].items()
            )
        else:
            end_captures = ()

        # Using the captures key for a begin/end rule is short-hand for
        # giving both beginCaptures and endCaptures with same values
        if begin and captures:
            end_captures = begin_captures = captures
            captures = ()

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


class FDict(Generic[TKey, TValue]):
    def __init__(self, dct: Dict[TKey, TValue]) -> None:
        self._hash = hash(tuple(sorted(dct.items())))
        self._dct = dct

    def __hash__(self) -> int:
        return self._hash

    def __getitem__(self, k: TKey) -> TValue:
        return self._dct[k]


class Grammar(NamedTuple):
    scope_name: str
    first_line_match: Optional[onigurumacffi._Pattern]
    file_types: FrozenSet[str]
    patterns: Tuple[_Rule, ...]
    repository: FDict[str, _Rule]

    @classmethod
    def parse(cls, filename: str) -> 'Grammar':
        with open(filename, 'rb') as f:
            # https://github.com/python/typeshed/pull/3738
            data = plistlib.load(f)  # type: ignore

        scope_name = data['scopeName']
        if 'firstLineMatch' in data:
            first_line_match = onigurumacffi.compile(data['firstLineMatch'])
        else:
            first_line_match = None
        if 'fileTypes' in data:
            file_types = frozenset(data['fileTypes'])
        else:
            file_types = frozenset()
        patterns = tuple(Rule.from_dct(dct) for dct in data['patterns'])
        if 'repository' in data:
            repository = FDict({
                k: Rule.from_dct(dct) for k, dct in data['repository'].items()
            })
        else:
            repository = FDict({})
        return cls(
            scope_name=scope_name,
            first_line_match=first_line_match,
            file_types=file_types,
            patterns=patterns,
            repository=repository,
        )

    @classmethod
    def blank(cls) -> 'Grammar':
        return cls(
            scope_name='source.unknown',
            first_line_match=None,
            file_types=frozenset(),
            patterns=(),
            repository=FDict({}),
        )

    def matches_file(self, filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        if ext.lstrip('.') in self.file_types:
            return True
        elif self.first_line_match is not None:
            with open(filename) as f:
                first_line = f.readline()
            return bool(self.first_line_match.match(first_line))
        else:
            return False


class Region(NamedTuple):
    start: int
    end: int
    scope: Scope


Regions = Tuple[Region, ...]
State = Tuple['_Entry', ...]
StyleCB = Callable[[Match[str], State], Tuple[State, Regions]]


class _Entry(Protocol):
    """hax for recursive types python/mypy#731"""
    @property
    def grammar(self) -> Grammar: ...
    @property
    def regset(self) -> onigurumacffi._RegSet: ...
    @property
    def callbacks(self) -> Tuple[StyleCB, ...]: ...
    @property
    def scope(self) -> Scope: ...


class Entry(NamedTuple):
    grammar: Grammar
    regset: onigurumacffi._RegSet
    callbacks: Tuple[StyleCB, ...]
    scope: Scope


@functools.lru_cache(maxsize=None)
def _highlight_line(state: State, line: str) -> Tuple[State, Regions]:
    ret = []
    pos = 0
    while pos < len(line):
        entry = state[-1]

        idx, match = entry.regset.search(line, pos)
        if match is not None:
            if match.start() > pos:
                ret.append(Region(pos, match.start(), entry.scope))

            state, regions = entry.callbacks[idx](match, state)
            ret.extend(regions)

            pos = match.end()
        else:
            ret.append(Region(pos, len(line), entry.scope))
            pos = len(line)

    return state, tuple(ret)


def _expand_captures(
        scope: Scope,
        match: Match[str],
        captures: Captures,
) -> Regions:
    ret: List[Region] = []
    pos, pos_end = match.span()
    for i, rule in captures:
        try:
            group_s = match[i]
        except IndexError:  # some grammars are malformed here?
            continue
        if not group_s:
            continue

        start, end = match.span(i)
        if start < pos:
            # TODO: could maybe bisect but this is probably fast enough
            j = len(ret) - 1
            while j > 0 and start < ret[j - 1].end:
                j -= 1

            oldtok = ret[j]
            newtok = []
            if start > oldtok.start:
                newtok.append(oldtok._replace(end=start))

            # TODO: this is duplicated below
            if not rule.match and not rule.begin and not rule.include:
                assert rule.name is not None
                newtok.append(Region(start, end, (*oldtok.scope, rule.name)))
            else:
                raise NotImplementedError('complex capture rule')

            if end < oldtok.end:
                newtok.append(oldtok._replace(start=end))
            ret[j:j + 1] = newtok
        else:
            if start > pos:
                ret.append(Region(pos, start, scope))

            if not rule.match and not rule.begin and not rule.include:
                assert rule.name is not None
                ret.append(Region(start, end, (*scope, rule.name)))
            else:
                raise NotImplementedError('complex capture rule')

            pos = end

    if pos < pos_end:
        ret.append(Region(pos, pos_end, scope))
    return tuple(ret)


def _end_cb(
        match: Match[str],
        state: State,
        *,
        end_captures: Captures,
) -> Tuple[State, Regions]:
    return state[:-1], _expand_captures(state[-1].scope, match, end_captures)


def _match_cb(
        match: Match[str],
        state: State,
        *,
        rule: _Rule,
) -> Tuple[State, Regions]:
    if rule.name is not None:
        scope = (*state[-1].scope, rule.name)
    else:
        scope = state[-1].scope
    return state, _expand_captures(scope, match, rule.captures)


def _begin_cb(
        match: Match[str],
        state: State,
        *,
        rule: _Rule,
) -> Tuple[State, Regions]:
    assert rule.end is not None
    prev_entry = state[-1]

    if rule.name is not None:
        scope = (*prev_entry.scope, rule.name)
    else:
        scope = prev_entry.scope
    if rule.content_name is not None:
        next_scopes = (*scope, rule.content_name)
    else:
        next_scopes = scope

    end = match.expand(rule.end)
    entry = _entry(
        prev_entry.grammar, rule.patterns, end, rule.end_captures, next_scopes,
    )

    return (*state, entry), _expand_captures(scope, match, rule.begin_captures)


@functools.lru_cache(maxsize=None)
def _regs_cbs(
        grammar: Grammar,
        rules: Tuple[_Rule, ...],
) -> Tuple[Tuple[str, ...], Tuple[StyleCB, ...]]:
    regs = []
    cbs: List[StyleCB] = []

    rules_stack = list(reversed(rules))
    while rules_stack:
        rule = rules_stack.pop()

        # XXX: can a rule have an include also?
        if rule.include is not None:
            assert rule.match is None
            assert rule.begin is None
            if rule.include == '$self':
                rules_stack.extend(reversed(grammar.patterns))
                continue
            else:
                rule = grammar.repository[rule.include[1:]]

        if rule.match is None and rule.begin is None and rule.patterns:
            rules_stack.extend(reversed(rule.patterns))
        elif rule.match is not None:
            regs.append(rule.match)
            cbs.append(functools.partial(_match_cb, rule=rule))
        elif rule.begin is not None:
            regs.append(rule.begin)
            cbs.append(functools.partial(_begin_cb, rule=rule))
        else:
            raise AssertionError(f'unreachable {rule}')

    return tuple(regs), tuple(cbs)


def _entry(
        grammar: Grammar,
        patterns: Tuple[_Rule, ...],
        end: str,
        end_captures: Captures,
        scope: Scope,
) -> _Entry:
    regs, cbs = _regs_cbs(grammar, patterns)
    end_cb = functools.partial(_end_cb, end_captures=end_captures)
    return Entry(grammar, compile_regset(end, *regs), (end_cb, *cbs), scope)


def _draw_screen(
        stdscr: 'curses._CursesWindow',
        curses_colors: Dict[Tuple[Color, Color], int],
        theme: Theme,
        grammar: Grammar,
        lines: List[str],
        y: int,
) -> None:
    regions_by_line = []

    entry = _entry(grammar, grammar.patterns, '$ ', (), (grammar.scope_name,))
    state: Tuple[_Entry, ...] = (entry,)
    for line in lines[:y + curses.LINES]:
        state, regions = _highlight_line(state, line)
        regions_by_line.append(regions)

    for i in range(curses.LINES):
        assert y + i < len(lines)
        stdscr.insstr(i, 0, lines[y + i])
        for start, end, scope in regions_by_line[y + i]:
            style = theme.select(scope)
            pair = curses_colors[(style.bg, style.fg)]
            stdscr.chgat(i, start, end - start, curses.color_pair(pair))


def _make_curses_colors(theme: Theme) -> Dict[Tuple[Color, Color], int]:
    assert curses.can_change_color()

    all_bgs = {theme.bg_default}.union(dict(theme.bg_rules).values())
    all_fgs = {theme.fg_default}.union(dict(theme.fg_rules).values())

    colors = {theme.bg_default: 0, theme.fg_default: 7}
    curses.init_color(0, *theme.bg_default.as_curses())
    curses.init_color(7, *theme.fg_default.as_curses())

    def _color_id() -> Generator[int, None, None]:
        """need to skip already assigned colors"""
        skip = frozenset(colors.values())
        i = 0
        while True:
            i += 1
            if i not in skip:
                yield i

    for i, color in zip(_color_id(), all_bgs | all_fgs):
        curses.init_color(i, *color.as_curses())
        colors[color] = i

    ret = {(theme.bg_default, theme.fg_default): 0}
    all_combinations = set(itertools.product(all_bgs, all_fgs))
    all_combinations.discard((theme.bg_default, theme.fg_default))
    for i, (bg, fg) in enumerate(all_combinations, 1):
        curses.init_pair(i, colors[fg], colors[bg])
        ret[(bg, fg)] = i

    return ret


def _highlight_curses(
        stdscr: 'curses._CursesWindow',
        theme: Theme,
        grammar: Grammar,
        filename: str,
) -> int:
    with open(filename) as f:
        lines = list(f)

    curses_colors = _make_curses_colors(theme)

    y = 0
    while True:
        _draw_screen(stdscr, curses_colors, theme, grammar, lines, y)

        wch = stdscr.get_wch()
        key = wch if isinstance(wch, int) else ord(wch)
        keyname = curses.keyname(key)
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
        elif key == curses.KEY_DOWN:
            y = min(y + 1, len(lines) - curses.LINES)
        elif key == curses.KEY_UP:
            y = max(0, y - 1)
        elif keyname == b'kEND5':
            y = len(lines) - curses.LINES
        elif keyname == b'kHOM5':
            y = 0
        elif wch == '"':
            lines[y] = '"' + lines[y]
        elif key == curses.KEY_DC:
            lines[y] = lines[y][1:]
        elif keyname == b'^X':
            break
        else:
            raise NotImplementedError(wch, key, keyname)
    return 0


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


def _highlight_output(theme: Theme, grammar: Grammar, filename: str) -> int:
    print(C_BG_TRUE.format(**theme.bg_default._asdict()))
    entry = _entry(grammar, grammar.patterns, '$ ', (), (grammar.scope_name,))
    state: Tuple[_Entry, ...] = (entry,)
    with open(filename) as f:
        for line in f:
            state, regions = _highlight_line(state, line)
            for start, end, scope in regions:
                print_styled(line[start:end], theme.select(scope))
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
    highlight_parser.add_argument(
        '--renderer',
        choices=('curses', 'output'),
        default='output',
    )
    highlight_parser.add_argument('theme')
    highlight_parser.add_argument('syntax_dir')
    highlight_parser.add_argument('filename')

    theme_parser = subparsers.add_parser('theme')
    theme_parser.add_argument('theme')

    args = parser.parse_args()

    for filename in os.listdir(args.syntax_dir):
        grammar = Grammar.parse(os.path.join(args.syntax_dir, filename))
        if grammar.matches_file(args.filename):
            break
    else:
        grammar = Grammar.blank()

    if args.command == 'highlight':
        theme = Theme.parse(args.theme)
        if args.renderer == 'curses':
            return curses.wrapper(
                _highlight_curses, theme, grammar, args.filename,
            )
        elif args.renderer == 'output':
            return _highlight_output(theme, grammar, args.filename)
        else:
            raise NotImplementedError(args.renderer)
    elif args.command == 'theme':
        return _theme(args.theme)
    else:
        raise NotImplementedError(args.command)


if __name__ == '__main__':
    exit(main())
