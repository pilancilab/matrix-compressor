#!/usr/bin/env python
import sys


def main():
    filename = sys.argv[1]
    if len(sys.argv) == 3:
        limit_lines = int(sys.argv[2])
    else:
        limit_lines = sys.maxsize
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == limit_lines:
                break
            if len(line) < 5:
                continue
            stripped_line = line.strip()
            if stripped_line.startswith("="):
                continue
            print(stripped_line)


if __name__ == "__main__":
    main()
