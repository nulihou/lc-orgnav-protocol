#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with open(p, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def main():
    if len(sys.argv) < 2:
        print("usage: check_jsonl_episode_count.py <jsonl_path>")
        sys.exit(2)
    p = Path(sys.argv[1])
    n = count_lines(p)
    ok = (n == 150)
    print(f"path={p.as_posix()} lines={n} expected=150 status={'OK' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()

