import argparse
import plistlib

import cson  # pip install cson


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('output')
    args = parser.parse_args()

    with open(args.filename) as src:
        contents = cson.load(src)

    with open(args.output, 'wb') as dest:
        plistlib.dump(contents, dest)

    return 0


if __name__ == '__main__':
    exit(main())
