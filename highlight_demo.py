import argparse
import contextlib
import functools
import itertools
import json
import os.path
import plistlib
import re
from typing import Any
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

compile_regex = functools.lru_cache()(onigurumacffi.compile)
compile_regset = functools.lru_cache()(onigurumacffi.compile_regset)

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')
Scope = Tuple[str, ...]
Regions = Tuple['Region', ...]
State = Tuple['Entry', ...]
Captures = Tuple[Tuple[int, '_Rule'], ...]
CapturesRef = Tuple[Tuple[int, int], ...]

C_256 = '\x1b[38;5;{c}m'
C_TRUE = '\x1b[38;2;{r};{g};{b}m'
C_BG_TRUE = '\x1b[48;2;{r};{g};{b}m'
C_RESET = '\x1b[m'

# yes I know this is wrong, but it's good enough for now
UN_COMMENT = re.compile(r'^\s*//.*$', re.MULTILINE)


class FDict(Generic[TKey, TValue]):
    def __init__(self, dct: Dict[TKey, TValue]) -> None:
        self._dct = dct

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


def _split_name(s: Optional[str]) -> Tuple[str, ...]:
    if s is None:
        return ()
    else:
        return tuple(s.split())


class _Rule(Protocol):
    """hax for recursive types python/mypy#731"""
    @property
    def name(self) -> Tuple[str, ...]: ...
    @property
    def match(self) -> Optional[str]: ...
    @property
    def begin(self) -> Optional[str]: ...
    @property
    def end(self) -> Optional[str]: ...
    @property
    def while_(self) -> Optional[str]: ...
    @property
    def content_name(self) -> Tuple[str, ...]: ...
    @property
    def captures(self) -> Captures: ...
    @property
    def begin_captures(self) -> Captures: ...
    @property
    def end_captures(self) -> Captures: ...
    @property
    def while_captures(self) -> Captures: ...
    @property
    def include(self) -> Optional[str]: ...
    @property
    def patterns(self) -> 'Tuple[_Rule, ...]': ...


class Rule(NamedTuple):
    name: Tuple[str, ...]
    match: Optional[str]
    begin: Optional[str]
    end: Optional[str]
    while_: Optional[str]
    content_name: Tuple[str, ...]
    captures: Captures
    begin_captures: Captures
    end_captures: Captures
    while_captures: Captures
    include: Optional[str]
    patterns: Tuple[_Rule, ...]

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> _Rule:
        name = _split_name(dct.get('name'))
        match = dct.get('match')
        begin = dct.get('begin')
        end = dct.get('end')
        while_ = dct.get('while')
        content_name = _split_name(dct.get('contentName'))

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

        if 'whileCaptures' in dct:
            while_captures = tuple(
                (int(k), Rule.from_dct(v))
                for k, v in dct['whileCaptures'].items()
            )
        else:
            while_captures = ()

        # Using the captures key for a begin/end/while rule is short-hand for
        # giving both beginCaptures and endCaptures with same values
        if begin and end and captures:
            begin_captures = end_captures = captures
            captures = ()
        elif begin and while_ and captures:
            begin_captures = while_captures = captures
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
            while_=while_,
            content_name=content_name,
            captures=captures,
            begin_captures=begin_captures,
            end_captures=end_captures,
            while_captures=while_captures,
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
            first_line_match = compile_regex(data['firstLineMatch'])
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


class CompiledRule(Protocol):
    @property
    def name(self) -> Tuple[str, ...]: ...

    def start(
            self,
            compiled: 'LazyGrammar',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        ...

    def search(
            self,
            compiled: 'LazyGrammar',
            state: State,
            line: str,
            pos: int,
    ) -> Tuple[State, int, Regions]:
        ...


class CompiledRegsetRule(CompiledRule, Protocol):
    @property
    def regset(self) -> onigurumacffi._RegSet: ...
    @property
    def rule_ids(self) -> Tuple[int, ...]: ...


class CompiledBeginRule(CompiledRule, Protocol):
    @property
    def content_name(self) -> Tuple[str, ...]: ...
    @property
    def begin_captures(self) -> CapturesRef: ...


class Entry(NamedTuple):
    scope: Tuple[str, ...]
    rule: CompiledRule
    reg: onigurumacffi._Pattern


def _inner_capture_parse(
        rules: 'LazyGrammar',
        start: int,
        s: str,
        scope: Scope,
        rule: CompiledRule,
) -> Regions:
    state = (Entry(scope + rule.name, rule, None),)
    _, regions = _highlight_line(rules, state, s)
    return tuple(
        r._replace(start=r.start + start, end=r.end + start) for r in regions
    )


def _captures(
        rules: 'LazyGrammar',
        scope: Scope,
        match: Match[str],
        captures: CapturesRef,
) -> Regions:
    ret: List[Region] = []
    pos, pos_end = match.span()
    for i, rule_id in captures:
        try:
            group_s = match[i]
        except IndexError:  # some grammars are malformed here?
            continue
        if not group_s:
            continue

        rule = rules[rule_id]
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

            newtok.extend(
                _inner_capture_parse(
                    rules, start, match[i], oldtok.scope, rule,
                ),
            )

            if end < oldtok.end:
                newtok.append(oldtok._replace(start=end))
            ret[j:j + 1] = newtok
        else:
            if start > pos:
                ret.append(Region(pos, start, scope))

            ret.extend(
                _inner_capture_parse(rules, start, match[i], scope, rule),
            )

            pos = end

    if pos < pos_end:
        ret.append(Region(pos, pos_end, scope))
    return tuple(ret)


def _do_regset(
        idx: int,
        match: Optional[Match[str]],
        rule: CompiledRegsetRule,
        rules: 'LazyGrammar',
        state: State,
        line: str,
        pos: int,
) -> Tuple[State, int, Regions]:
    ret = []
    if match is not None:
        if match.start() > pos:
            ret.append(Region(pos, match.start(), state[-1].scope))

        state, regions = rules[rule.rule_ids[idx]].start(rules, match, state)

        ret.extend(regions)
        pos = match.end()
    else:
        ret.append(Region(pos, len(line), state[-1].scope))
        pos = len(line)
    return state, pos, tuple(ret)


class PatternRule(NamedTuple):
    name: Tuple[str, ...]
    regset: onigurumacffi._RegSet
    rule_ids: Tuple[int, ...]

    def start(
            self,
            rules: 'LazyGrammar',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        raise AssertionError(f'unreachable {self}')

    def search(
            self,
            rules: 'LazyGrammar',
            state: State,
            line: str,
            pos: int,
    ) -> Tuple[State, int, Regions]:
        idx, match = self.regset.search(line, pos)
        return _do_regset(idx, match, self, rules, state, line, pos)


class MatchRule(NamedTuple):
    name: Tuple[str, ...]
    captures: CapturesRef

    def start(
            self,
            rules: 'LazyGrammar',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        scope = state[-1].scope + self.name
        return state, _captures(rules, scope, match, self.captures)

    def search(
            self,
            rules: 'LazyGrammar',
            state: State,
            line: str,
            pos: int,
    ) -> Tuple[State, int, Regions]:
        raise AssertionError(f'unreachable {self}')


def _begin_start(
        rule: CompiledBeginRule,
        rules: 'LazyGrammar',
        match: Match[str],
        state: State,
        regstr: str,
) -> Tuple[State, Regions]:
    scope = state[-1].scope + rule.name
    next_scope = scope + rule.content_name

    compiled_regex = compile_regex(match.expand(regstr))
    state = (*state, Entry(next_scope, rule, compiled_regex))
    return state, _captures(rules, scope, match, rule.begin_captures)


class BeginEndRule(NamedTuple):
    name: Tuple[str, ...]
    content_name: Tuple[str, ...]
    begin_captures: CapturesRef
    end_captures: CapturesRef
    begin: onigurumacffi._Pattern
    end: str
    regset: onigurumacffi._RegSet
    rule_ids: Tuple[int, ...]

    def start(
            self,
            rules: 'LazyGrammar',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        return _begin_start(self, rules, match, state, self.end)

    def search(
            self,
            rules: 'LazyGrammar',
            state: State,
            line: str,
            pos: int,
    ) -> Tuple[State, int, Regions]:
        def _end_ret() -> Tuple[State, int, Regions]:
            ret = []
            if end_match.start() > pos:
                ret.append(Region(pos, end_match.start(), state[-1].scope))
            ret.extend(
                _captures(
                    rules, state[-1].scope, end_match, self.end_captures,
                ),
            )
            return state[:-1], end_match.end(), tuple(ret)

        end_match = state[-1].reg.search(line, pos)
        if end_match is not None and end_match.start() == pos:
            return _end_ret()
        elif end_match is None:
            idx, match = self.regset.search(line, pos)
            return _do_regset(idx, match, self, rules, state, line, pos)
        else:
            idx, match = self.regset.search(line, pos)
            if match is None or end_match.start() < match.start():
                return _end_ret()
            else:
                return _do_regset(idx, match, self, rules, state, line, pos)


class BeginWhileRule(NamedTuple):
    name: Tuple[str, ...]
    content_name: Tuple[str, ...]
    begin_captures: CapturesRef
    while_captures: CapturesRef
    begin: onigurumacffi._Pattern
    while_: str
    regset: onigurumacffi._RegSet
    rule_ids: Tuple[int, ...]

    def start(
            self,
            rules: 'LazyGrammar',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        return _begin_start(self, rules, match, state, self.while_)

    def search(
            self,
            rules: 'LazyGrammar',
            state: State,
            line: str,
            pos: int,
    ) -> Tuple[State, int, Regions]:
        ret: Regions = ()
        if pos == 0:
            while_match = state[-1].reg.match(line)
            if while_match is None:
                return state[:-1], pos, ()
            else:
                ret = _captures(
                    rules, state[-1].scope, while_match, self.while_captures,
                )
                pos = while_match.end()

        idx, match = self.regset.search(line, pos)
        regset_ret = _do_regset(idx, match, self, rules, state, line, pos)
        state, pos, regions = regset_ret
        return state, pos, ret + regions


@functools.lru_cache(maxsize=None)
def _expand_include(
        grammar: Grammar,
        s: str,
) -> Tuple[List[str], List[_Rule]]:
    if s == '$self':
        return _expand_patterns(grammar, grammar.patterns)
    else:
        return _expand_patterns(grammar, (grammar.repository[s[1:]],))


@functools.lru_cache(maxsize=None)
def _expand_patterns(
        grammar: Grammar,
        rules: Tuple[_Rule, ...],
) -> Tuple[List[str], List[_Rule]]:
    ret_regs, ret_rules = [], []
    for rule in rules:
        if rule.include is not None:
            inner_regs, inner_rules = _expand_include(grammar, rule.include)
            ret_regs.extend(inner_regs)
            ret_rules.extend(inner_rules)
        elif rule.match is None and rule.begin is None and rule.patterns:
            inner_regs, inner_rules = _expand_patterns(grammar, rule.patterns)
            ret_regs.extend(inner_regs)
            ret_rules.extend(inner_rules)
        elif rule.match is not None:
            ret_regs.append(rule.match)
            ret_rules.append(rule)
        elif rule.begin is not None:
            ret_regs.append(rule.begin)
            ret_rules.append(rule)
        else:
            raise AssertionError(f'unreachable {rule}')
    return ret_regs, ret_rules


def _captures_ref(captures: Captures) -> Tuple[List[_Rule], CapturesRef]:
    rules = [rule for _, rule in captures]
    captures_ref = tuple((n, id(rule)) for n, rule in captures)
    return rules, captures_ref


def _compile_root(grammar: Grammar) -> Tuple[PatternRule, List[_Rule]]:
    regs, rules = _expand_patterns(grammar, grammar.patterns)
    compiled = PatternRule(
        (grammar.scope_name,),
        compile_regset(*regs),
        tuple(id(rule) for rule in rules),
    )
    return compiled, rules


def _compile_rule(
        grammar: Grammar,
        rule: _Rule,
) -> Tuple[CompiledRule, List[_Rule]]:
    assert rule.include is None, rule
    if rule.match is not None:
        rules, captures_ref = _captures_ref(rule.captures)
        return MatchRule(rule.name, captures_ref), rules
    elif rule.begin is not None and rule.end is not None:
        regs, rules = _expand_patterns(grammar, rule.patterns)
        begin_rules, begin_captures_ref = _captures_ref(rule.begin_captures)
        end_rules, end_captures_ref = _captures_ref(rule.end_captures)
        begin_end_rule = BeginEndRule(
            rule.name,
            rule.content_name,
            begin_captures_ref,
            end_captures_ref,
            compile_regex(rule.begin),
            rule.end,
            compile_regset(*regs),
            tuple(id(rule) for rule in rules),
        )
        return begin_end_rule, rules + begin_rules + end_rules
    elif rule.begin is not None and rule.while_ is not None:
        regs, rules = _expand_patterns(grammar, rule.patterns)
        begin_rules, begin_captures_ref = _captures_ref(rule.begin_captures)
        while_rules, while_captures_ref = _captures_ref(rule.while_captures)
        begin_while_rule = BeginWhileRule(
            rule.name,
            rule.content_name,
            begin_captures_ref,
            while_captures_ref,
            compile_regex(rule.begin),
            rule.while_,
            compile_regset(*regs),
            tuple(id(rule) for rule in rules),
        )
        return begin_while_rule, rules + begin_rules + while_rules
    else:
        regs, rules = _expand_patterns(grammar, rule.patterns)
        compiled = PatternRule(
            rule.name,
            compile_regset(*regs),
            tuple(id(rule) for rule in rules),
        )
        return compiled, rules


class LazyGrammar:
    def __init__(self, grammar: Grammar) -> None:
        self._grammar = grammar
        self.root, to_parse = _compile_root(grammar)
        self._c_rules: Dict[int, CompiledRule] = {id(grammar): self.root}
        self._u_rules = {id(rule): rule for rule in to_parse}

    def __getitem__(self, rule_id: int) -> CompiledRule:
        with contextlib.suppress(KeyError):
            return self._c_rules[rule_id]

        ret, to_parse = _compile_rule(self._grammar, self._u_rules[rule_id])
        self._c_rules[rule_id] = ret
        for rule in to_parse:
            self._u_rules[id(rule)] = rule
        return ret


@functools.lru_cache(maxsize=None)
def _highlight_line(
        rules: 'LazyGrammar',
        state: State,
        line: str,
) -> Tuple[State, Regions]:
    ret: List[Region] = []
    pos = 0
    while pos < len(line):
        state, pos, regions = state[-1].rule.search(rules, state, line, pos)
        ret.extend(regions)

    return state, tuple(ret)


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
    rules = LazyGrammar(grammar)
    state: State = (Entry(rules.root.name, rules.root, None),)

    print(C_BG_TRUE.format(**theme.default.bg._asdict()))
    with open(filename) as f:
        for line in f:
            state, regions = _highlight_line(rules, state, line)
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
