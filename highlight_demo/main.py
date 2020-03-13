import argparse
import os.path

from highlight_demo.highlight import Compiler
from highlight_demo.highlight import Grammars
from highlight_demo.highlight import highlight_line
from highlight_demo.theme import Style
from highlight_demo.theme import Theme

HERE = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_THEME = os.path.join(HERE, 'demo/themes/dark_plus_vs.json')
DEFAULT_SYNTAX_DIR = os.path.join(HERE, 'demo/languages')


def print_styled(s: str, style: Style) -> None:
    color_s = ''
    undo_s = ''
    color_s += '\x1b[38;2;{r};{g};{b}m'.format(**style.fg._asdict())
    color_s += '\x1b[48;2;{r};{g};{b}m'.format(**style.bg._asdict())
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
    state = compiler.root_state

    print('\x1b[48;2;{r};{g};{b}m'.format(**theme.default.bg._asdict()))
    with open(filename) as f:
        for line_idx, line in enumerate(f):
            first_line = line_idx == 0
            state, regions = highlight_line(compiler, state, line, first_line)
            for start, end, scope in regions:
                print_styled(line[start:end], theme.select(scope))
    print('\x1b[m', end='')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default=DEFAULT_THEME)
    parser.add_argument('--syntax-dir', default=DEFAULT_SYNTAX_DIR)
    parser.add_argument('filename')

    args = parser.parse_args()

    theme = Theme.from_filename(args.theme)

    grammars = Grammars.from_syntax_dir(args.syntax_dir)
    compiler = grammars.compiler_for_file(args.filename)

    return _highlight_output(theme, compiler, args.filename)


if __name__ == '__main__':
    exit(main())
