import functools
import json
import os.path
import re
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

from highlight_demo._types import Protocol
from highlight_demo.color import Color
from highlight_demo.fdict import FDict

# yes I know this is wrong, but it's good enough for now
UN_COMMENT = re.compile(r'^\s*//.*$', re.MULTILINE)


class Style(NamedTuple):
    fg: Optional[Color]
    bg: Optional[Color]
    b: bool
    i: bool
    u: bool

    @classmethod
    def blank(cls) -> 'Style':
        return cls(fg=None, bg=None, b=False, i=False, u=False)


class PartialStyle(NamedTuple):
    fg: Optional[Color] = None
    bg: Optional[Color] = None
    b: Optional[bool] = None
    i: Optional[bool] = None
    u: Optional[bool] = None

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


class _TrieNode(Protocol):
    @property
    def style(self) -> PartialStyle: ...
    @property
    def children(self) -> FDict[str, '_TrieNode']: ...


class TrieNode(NamedTuple):
    style: PartialStyle
    children: FDict[str, _TrieNode]

    @classmethod
    def from_dct(cls, dct: Dict[str, Any]) -> _TrieNode:
        children = FDict({
            k: TrieNode.from_dct(v) for k, v in dct['children'].items()
        })
        return cls(PartialStyle.from_dct(dct), children)


class Theme(NamedTuple):
    default: Style
    rules: _TrieNode

    @functools.lru_cache(maxsize=None)
    def select(self, scope: Tuple[str, ...]) -> Style:
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
        default = Style.blank()._asdict()

        for k in ('foreground', 'editor.foreground'):
            if k in data.get('colors', {}):
                default['fg'] = Color.parse(data['colors'][k])
                break

        for k in ('background', 'editor.background'):
            if k in data.get('colors', {}):
                default['bg'] = Color.parse(data['colors'][k])
                break

        root: Dict[str, Any] = {'children': {}}
        rules = data.get('tokenColors', []) + data.get('settings', [])
        for rule in rules:
            if 'scope' not in rule:
                scopes = ['']
            elif isinstance(rule['scope'], str):
                scopes = [
                    s.strip()
                    # some themes have a buggy trailing/leading comma
                    for s in rule['scope'].strip().strip(',').split(',')
                ]
            else:
                scopes = rule['scope']

            for scope in scopes:
                if ' ' in scope:
                    # TODO: implement parent scopes
                    continue
                elif scope == '':
                    PartialStyle.from_dct(rule['settings']).overlay_on(default)
                    continue

                cur = root
                for part in scope.split('.'):
                    cur = cur['children'].setdefault(part, {'children': {}})

                cur.update(rule['settings'])

        return cls(Style(**default), TrieNode.from_dct(root))

    @classmethod
    def blank(cls) -> 'Theme':
        return cls(Style.blank(), TrieNode.from_dct({'children': {}}))

    @classmethod
    def from_filename(cls, filename: str) -> 'Theme':
        if not os.path.exists(filename):
            return cls.blank()
        else:
            with open(filename) as f:
                contents = UN_COMMENT.sub('', f.read())
                return cls.from_dct(json.loads(contents))
