import argparse
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


class FDict(Generic[TKey, TValue]):
    def __init__(self, dct: Dict[TKey, TValue]) -> None:
        self._hash = hash(tuple(sorted(dct.items())))
        self._dct = dct

    def __hash__(self) -> int:
        return self._hash

    def __getitem__(self, k: TKey) -> TValue:
        return self._dct[k]

    def __contains__(self, k: TKey) -> bool:
        return k in self._dct


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

    @classmethod
    def blank(cls) -> 'Style':
        return cls(
            fg=Color(0xff, 0xff, 0xff), bg=Color(0x00, 0x00, 0x00),
            b=False, u=False, i=False,
        )


class PartialStyle(NamedTuple):
    fg: Optional[Color] = None
    bg: Optional[Color] = None
    b: Optional[bool] = None
    u: Optional[bool] = None
    i: Optional[bool] = None

    def overlay_on(self, dct: Dict[str, Any]) -> None:
        for attr in self._fields:
            value = getattr(self, attr)
            if value is not None:
                dct[attr] = value

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> 'PartialStyle':
        kv = cls()._asdict()
        if 'foreground' in dct:
            kv['fg'] = Color.parse(dct['foreground'])
        if 'background' in dct:
            kv['bg'] = Color.parse(dct['background'])
        if dct.get('fontStyle') == 'bold':
            kv['b'] = True
        elif dct.get('fontStyle') == 'italic':
            kv['i'] = True
        elif dct.get('fontStyle') == 'underline':
            kv['u'] = True
        return cls(**kv)


class _ThemeTrieNode(Protocol):
    @property
    def style(self) -> PartialStyle: ...
    @property
    def children(self) -> FDict[str, '_ThemeTrieNode']: ...


class ThemeTrieNode(NamedTuple):
    style: PartialStyle
    children: FDict[str, _ThemeTrieNode]

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> _ThemeTrieNode:
        children = FDict({
            k: ThemeTrieNode.from_dct(v) for k, v in dct['children'].items()
        })
        return cls(PartialStyle.from_dct(dct), children)


class Theme(NamedTuple):
    default: Style
    rules: _ThemeTrieNode

    @functools.lru_cache(maxsize=None)
    def select(self, scope: Scope) -> Style:
        if not scope:
            return self.default
        else:
            style = self.select(scope[:-1])._asdict()
            node = self.rules
            for part in scope[-1].split('.'):
                if part not in node.children:
                    break
                else:
                    node = node.children[part]
                    node.style.overlay_on(style)
            return Style(**style)

    @classmethod
    def from_dct(cls, data: Dict[str, Any]) -> 'Theme':
        root: Dict[str, Any] = {'children': {}}

        default = Style.blank()._asdict()

        for k in ('foreground', 'editor.foreground'):
            if k in data['colors']:
                default['fg'] = Color.parse(data['colors'][k])
                break

        for k in ('background', 'editor.background'):
            if k in data['colors']:
                default['bg'] = Color.parse(data['colors'][k])
                break

        for rule in data['tokenColors']:
            if 'scope' not in rule:
                scopes = ['']
            elif isinstance(rule['scope'], str):
                scopes = [
                    s.strip() for s in rule['scope'].split(',')
                    # some themes have a buggy trailing comma
                    if s.strip()
                ]
            else:
                scopes = rule['scope']

            for scope in scopes:
                if ' ' in scope:
                    # TODO: implement parent scopes
                    continue
                elif scope == '':
                    PartialStyle.from_dct(rule['settings']).overlay_on(default)

                cur = root
                for part in scope.split('.'):
                    cur = cur['children'].setdefault(part, {'children': {}})

                cur.update(rule['settings'])

        return cls(Style(**default), ThemeTrieNode.from_dct(root))

    @classmethod
    def parse(cls, filename: str) -> 'Theme':
        with open(filename) as f:
            contents = UN_COMMENT.sub('', f.read())
            return cls.from_dct(json.loads(contents))


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
    print(C_BG_TRUE.format(**theme.default.bg._asdict()))
    entry = _entry(grammar, grammar.patterns, '$ ', (), (grammar.scope_name,))
    state: Tuple[_Entry, ...] = (entry,)
    with open(filename) as f:
        for line in f:
            state, regions = _highlight_line(state, line)
            for start, end, scope in regions:
                print_styled(line[start:end], theme.select(scope))
    print('\x1b[m', end='')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('theme')
    parser.add_argument('syntax_dir')
    parser.add_argument('filename')

    args = parser.parse_args()

    for filename in os.listdir(args.syntax_dir):
        grammar = Grammar.parse(os.path.join(args.syntax_dir, filename))
        if grammar.matches_file(args.filename):
            break
    else:
        grammar = Grammar.blank()

    theme = Theme.parse(args.theme)
    return _highlight_output(theme, grammar, args.filename)


if __name__ == '__main__':
    exit(main())
