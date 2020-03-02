import argparse
import contextlib
import functools
import itertools
import json
import os.path
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
Captures = Tuple[Tuple[int, '_Rule'], ...]

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
        with open(filename) as f:
            data = json.load(f)

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

    def matches_file(self, filename: str, first_line: str) -> bool:
        _, ext = os.path.splitext(filename)
        if ext.lstrip('.') in self.file_types:
            return True
        elif self.first_line_match is not None:
            return bool(self.first_line_match.match(first_line))
        else:
            return False


class Region(NamedTuple):
    start: int
    end: int
    scope: Scope


class State(NamedTuple):
    entries: Tuple['Entry', ...]
    while_stack: Tuple[Tuple['WhileRule', int], ...]

    @classmethod
    def root(cls, entry: 'Entry') -> 'State':
        return cls((entry,), ())

    @property
    def cur(self) -> 'Entry':
        return self.entries[-1]

    def push(self, entry: 'Entry') -> 'State':
        return self._replace(entries=(*self.entries, entry))

    def pop(self) -> 'State':
        return self._replace(entries=self.entries[:-1])

    def push_while(self, rule: 'WhileRule', entry: 'Entry') -> 'State':
        entries = (*self.entries, entry)
        while_stack = (*self.while_stack, (rule, len(entries)))
        return self._replace(entries=entries, while_stack=while_stack)

    def pop_while(self) -> 'State':
        entries, while_stack = self.entries[:-1], self.while_stack[:-1]
        return self._replace(entries=entries, while_stack=while_stack)


class CompiledRule(Protocol):
    @property
    def name(self) -> Tuple[str, ...]: ...

    def start(
            self,
            compiler: 'Compiler',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        ...

    def search(
            self,
            compiler: 'Compiler',
            state: State,
            line: str,
            pos: int,
    ) -> Optional[Tuple[State, int, Regions]]:
        ...


class CompiledRegsetRule(CompiledRule, Protocol):
    @property
    def regset(self) -> onigurumacffi._RegSet: ...
    @property
    def u_rules(self) -> Tuple[_Rule, ...]: ...


class Entry(NamedTuple):
    scope: Tuple[str, ...]
    rule: CompiledRule
    reg: onigurumacffi._Pattern


def _inner_capture_parse(
        compiler: 'Compiler',
        start: int,
        s: str,
        scope: Scope,
        rule: CompiledRule,
) -> Regions:
    state = State.root(Entry(scope + rule.name, rule, None))
    _, regions = highlight_line(compiler, state, s)
    return tuple(
        r._replace(start=r.start + start, end=r.end + start) for r in regions
    )


def _captures(
        compiler: 'Compiler',
        scope: Scope,
        match: Match[str],
        captures: Captures,
) -> Regions:
    ret: List[Region] = []
    pos, pos_end = match.span()
    for i, u_rule in captures:
        try:
            group_s = match[i]
        except IndexError:  # some grammars are malformed here?
            continue
        if not group_s:
            continue

        rule = compiler.compile_rule(u_rule)
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
                    compiler, start, match[i], oldtok.scope, rule,
                ),
            )

            if end < oldtok.end:
                newtok.append(oldtok._replace(start=end))
            ret[j:j + 1] = newtok
        else:
            if start > pos:
                ret.append(Region(pos, start, scope))

            ret.extend(
                _inner_capture_parse(compiler, start, match[i], scope, rule),
            )

            pos = end

    if pos < pos_end:
        ret.append(Region(pos, pos_end, scope))
    return tuple(ret)


def _do_regset(
        idx: int,
        match: Optional[Match[str]],
        rule: CompiledRegsetRule,
        compiler: 'Compiler',
        state: State,
        pos: int,
) -> Optional[Tuple[State, int, Regions]]:
    if match is None:
        return None

    ret = []
    if match.start() > pos:
        ret.append(Region(pos, match.start(), state.cur.scope))

    target_rule = compiler.compile_rule(rule.u_rules[idx])
    state, regions = target_rule.start(compiler, match, state)
    ret.extend(regions)

    return state, match.end(), tuple(ret)


class PatternRule(NamedTuple):
    name: Tuple[str, ...]
    regset: onigurumacffi._RegSet
    u_rules: Tuple[_Rule, ...]

    def start(
            self,
            compiler: 'Compiler',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        raise AssertionError(f'unreachable {self}')

    def search(
            self,
            compiler: 'Compiler',
            state: State,
            line: str,
            pos: int,
    ) -> Optional[Tuple[State, int, Regions]]:
        idx, match = self.regset.search(line, pos)
        return _do_regset(idx, match, self, compiler, state, pos)


class MatchRule(NamedTuple):
    name: Tuple[str, ...]
    captures: Captures

    def start(
            self,
            compiler: 'Compiler',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        scope = state.cur.scope + self.name
        return state, _captures(compiler, scope, match, self.captures)

    def search(
            self,
            compiler: 'Compiler',
            state: State,
            line: str,
            pos: int,
    ) -> Optional[Tuple[State, int, Regions]]:
        raise AssertionError(f'unreachable {self}')


class EndRule(NamedTuple):
    name: Tuple[str, ...]
    content_name: Tuple[str, ...]
    begin_captures: Captures
    end_captures: Captures
    end: str
    regset: onigurumacffi._RegSet
    u_rules: Tuple[_Rule, ...]

    def start(
            self,
            compiler: 'Compiler',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        scope = state.cur.scope + self.name
        next_scope = scope + self.content_name

        end = compile_regex(match.expand(self.end))
        state = state.push(Entry(next_scope, self, end))
        return state, _captures(compiler, scope, match, self.begin_captures)

    def search(
            self,
            compiler: 'Compiler',
            state: State,
            line: str,
            pos: int,
    ) -> Optional[Tuple[State, int, Regions]]:
        def _end_ret() -> Tuple[State, int, Regions]:
            ret = []
            if end_match.start() > pos:
                ret.append(Region(pos, end_match.start(), state.cur.scope))
            ret.extend(
                _captures(
                    compiler, state.cur.scope, end_match, self.end_captures,
                ),
            )
            return state.pop(), end_match.end(), tuple(ret)

        end_match = state.cur.reg.search(line, pos)
        if end_match is not None and end_match.start() == pos:
            return _end_ret()
        elif end_match is None:
            idx, match = self.regset.search(line, pos)
            return _do_regset(idx, match, self, compiler, state, pos)
        else:
            idx, match = self.regset.search(line, pos)
            if match is None or end_match.start() < match.start():
                return _end_ret()
            else:
                return _do_regset(idx, match, self, compiler, state, pos)


class WhileRule(NamedTuple):
    name: Tuple[str, ...]
    content_name: Tuple[str, ...]
    begin_captures: Captures
    while_captures: Captures
    while_: str
    regset: onigurumacffi._RegSet
    u_rules: Tuple[_Rule, ...]

    def start(
            self,
            compiler: 'Compiler',
            match: Match[str],
            state: State,
    ) -> Tuple[State, Regions]:
        scope = state.cur.scope + self.name
        next_scope = scope + self.content_name

        while_ = compile_regex(match.expand(self.while_))
        state = state.push_while(self, Entry(next_scope, self, while_))
        return state, _captures(compiler, scope, match, self.begin_captures)

    def continues(
            self,
            compiler: 'Compiler',
            state: State,
            line: str,
            pos: int,
    ) -> Optional[Tuple[int, Regions]]:
        match = state.cur.reg.match(line, pos)
        if match is None:
            return None

        ret = _captures(compiler, state.cur.scope, match, self.while_captures)
        return match.end(), ret

    def search(
            self,
            compiler: 'Compiler',
            state: State,
            line: str,
            pos: int,
    ) -> Optional[Tuple[State, int, Regions]]:
        idx, match = self.regset.search(line, pos)
        return _do_regset(idx, match, self, compiler, state, pos)


class Compiler:
    def __init__(self, grammar: Grammar, grammars: Dict[str, Grammar]) -> None:
        self._root_scope = grammar.scope_name
        self._grammars = grammars
        self._rule_to_grammar: Dict[_Rule, Grammar] = {}
        self._c_rules: Dict[_Rule, CompiledRule] = {}
        self.root = self._compile_root(grammar)

    def _visit_rule(self, grammar: Grammar, rule: _Rule) -> _Rule:
        self._rule_to_grammar[rule] = grammar
        return rule

    @functools.lru_cache(maxsize=None)
    def _include(
            self,
            grammar: Grammar,
            s: str,
    ) -> Tuple[List[str], Tuple[_Rule, ...]]:
        if s == '$self':
            return self._patterns(grammar, grammar.patterns)
        elif s == '$base':
            return self._include(self._grammars[self._root_scope], '$self')
        elif s.startswith('#'):
            return self._patterns(grammar, (grammar.repository[s[1:]],))
        elif '#' not in s:
            return self._include(self._grammars[s], '$self')
        else:
            scope, _, s = s.partition('#')
            return self._include(self._grammars[scope], f'#{s}')

    @functools.lru_cache(maxsize=None)
    def _patterns(
            self,
            grammar: Grammar,
            rules: Tuple[_Rule, ...],
    ) -> Tuple[List[str], Tuple[_Rule, ...]]:
        ret_regs = []
        ret_rules: List[_Rule] = []
        for rule in rules:
            if rule.include is not None:
                tmp_regs, tmp_rules = self._include(grammar, rule.include)
                ret_regs.extend(tmp_regs)
                ret_rules.extend(tmp_rules)
            elif rule.match is None and rule.begin is None and rule.patterns:
                tmp_regs, tmp_rules = self._patterns(grammar, rule.patterns)
                ret_regs.extend(tmp_regs)
                ret_rules.extend(tmp_rules)
            elif rule.match is not None:
                ret_regs.append(rule.match)
                ret_rules.append(self._visit_rule(grammar, rule))
            elif rule.begin is not None:
                ret_regs.append(rule.begin)
                ret_rules.append(self._visit_rule(grammar, rule))
            else:
                raise AssertionError(f'unreachable {rule}')
        return ret_regs, tuple(ret_rules)

    def _captures_ref(
            self,
            grammar: Grammar,
            captures: Captures,
    ) -> Captures:
        return tuple((n, self._visit_rule(grammar, r)) for n, r in captures)

    def _compile_root(self, grammar: Grammar) -> PatternRule:
        regs, rules = self._patterns(grammar, grammar.patterns)
        return PatternRule((grammar.scope_name,), compile_regset(*regs), rules)

    def _compile_rule(self, grammar: Grammar, rule: _Rule) -> CompiledRule:
        assert rule.include is None, rule
        if rule.match is not None:
            captures_ref = self._captures_ref(grammar, rule.captures)
            return MatchRule(rule.name, captures_ref)
        elif rule.begin is not None and rule.end is not None:
            regs, rules = self._patterns(grammar, rule.patterns)
            return EndRule(
                rule.name,
                rule.content_name,
                self._captures_ref(grammar, rule.begin_captures),
                self._captures_ref(grammar, rule.end_captures),
                rule.end,
                compile_regset(*regs),
                rules,
            )
        elif rule.begin is not None and rule.while_ is not None:
            regs, rules = self._patterns(grammar, rule.patterns)
            return WhileRule(
                rule.name,
                rule.content_name,
                self._captures_ref(grammar, rule.begin_captures),
                self._captures_ref(grammar, rule.while_captures),
                rule.while_,
                compile_regset(*regs),
                rules,
            )
        else:
            regs, rules = self._patterns(grammar, rule.patterns)
            return PatternRule(rule.name, compile_regset(*regs), rules)

    def compile_rule(self, rule: _Rule) -> CompiledRule:
        with contextlib.suppress(KeyError):
            return self._c_rules[rule]

        grammar = self._rule_to_grammar[rule]
        ret = self._c_rules[rule] = self._compile_rule(grammar, rule)
        return ret


class Grammars:
    def __init__(self, grammars: List[Grammar]) -> None:
        self.grammars = {grammar.scope_name: grammar for grammar in grammars}
        self._compilers: Dict[Grammar, Compiler] = {}

    @classmethod
    def from_syntax_dir(cls, syntax_dir: str) -> 'Grammars':
        grammars = [Grammar.blank()]
        if os.path.exists(syntax_dir):
            grammars.extend(
                Grammar.parse(os.path.join(syntax_dir, filename))
                for filename in os.listdir(syntax_dir)
            )
        return cls(grammars)

    def _compiler_for_grammar(self, grammar: Grammar) -> Compiler:
        with contextlib.suppress(KeyError):
            return self._compilers[grammar]

        ret = self._compilers[grammar] = Compiler(grammar, self.grammars)
        return ret

    def blank_compiler(self) -> Compiler:
        grammar = self.grammars['source.unknown']
        return self._compiler_for_grammar(grammar)

    def compiler_for_file(self, filename: str) -> Compiler:
        if os.path.exists(filename):
            with open(filename) as f:
                first_line = next(f)
        else:
            first_line = ''
        for grammar in self.grammars.values():
            if grammar.matches_file(filename, first_line):
                break
        else:
            grammar = self.grammars['source.unknown']

        return self._compiler_for_grammar(grammar)


@functools.lru_cache(maxsize=None)
def highlight_line(
        compiler: 'Compiler',
        state: State,
        line: str,
) -> Tuple[State, Regions]:
    ret: List[Region] = []
    pos = 0

    # TODO: this is still a little wasteful
    while_stack = []
    for while_rule, idx in state.while_stack:
        while_stack.append((while_rule, idx))
        while_state = State(state.entries[:idx], tuple(while_stack))

        while_res = while_rule.continues(compiler, while_state, line, pos)
        if while_res is None:
            state = while_state.pop_while()
            break
        else:
            pos, regions = while_res
            ret.extend(regions)

    search_res = state.cur.rule.search(compiler, state, line, pos)
    while search_res is not None:
        state, pos, regions = search_res
        ret.extend(regions)

        search_res = state.cur.rule.search(compiler, state, line, pos)

    if pos < len(line):
        ret.append(Region(pos, len(line), state.cur.scope))

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
    print(f'{color_s}{s}{undo_s}', end='', flush=True)


def _highlight_output(theme: Theme, compiler: Compiler, filename: str) -> int:
    state = State.root(Entry(compiler.root.name, compiler.root, None))

    print(C_BG_TRUE.format(**theme.default.bg._asdict()))
    with open(filename) as f:
        for line in f:
            state, regions = highlight_line(compiler, state, line)
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

    theme = Theme.parse(args.theme)

    grammars = Grammars.from_syntax_dir(args.syntax_dir)
    compiler = grammars.compiler_for_file(args.filename)

    return _highlight_output(theme, compiler, args.filename)


if __name__ == '__main__':
    exit(main())
