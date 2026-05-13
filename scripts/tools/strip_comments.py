from __future__ import annotations

import sys
import tokenize
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git"}


def should_skip(p: Path) -> bool:
    return any(part in SKIP_DIRS for part in p.parts)


def strip_py_comments(src: str) -> str:
    lines = src.splitlines(keepends=True)
    if not lines:
        return src

    try:
        tokens = list(tokenize.generate_tokens(StringIO(src).readline))
    except tokenize.TokenizeError:
        return src

    comments_by_line: dict[int, int] = {}
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        if row == 1 and tok.string.startswith("#!"):
            continue
        if row not in comments_by_line or col < comments_by_line[row]:
            comments_by_line[row] = col

    new_lines = []
    for i, line in enumerate(lines):
        row = i + 1
        if row in comments_by_line:
            col = comments_by_line[row]
            head = line[:col]
            trailing_nl = "\n" if line.endswith("\n") else ""
            stripped_head = head.rstrip()
            if stripped_head == "":
                new_lines.append(trailing_nl)
            else:
                new_lines.append(stripped_head + trailing_nl)
        else:
            new_lines.append(line)
    return "".join(new_lines)


def strip_hash_comments_linewise(src: str) -> str:
    out = []
    for i, line in enumerate(src.splitlines(keepends=True)):
        if i == 0 and line.startswith("#!"):
            out.append(line)
            continue
        in_single = False
        in_double = False
        cut = None
        for j, ch in enumerate(line):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "#" and not in_single and not in_double:
                cut = j
                break
        if cut is None:
            out.append(line)
            continue
        head = line[:cut]
        trailing_nl = "\n" if line.endswith("\n") else ""
        stripped = head.rstrip()
        if stripped == "":
            out.append(trailing_nl)
        else:
            out.append(stripped + trailing_nl)
    return "".join(out)


def gather_files() -> list[Path]:
    files = []
    for ext in ("*.py", "*.sh", "*.toml"):
        for p in ROOT.rglob(ext):
            if should_skip(p):
                continue
            if p.resolve() == Path(__file__).resolve():
                continue
            files.append(p)
    return sorted(files)


def main(dry_run: bool = False) -> None:
    files = gather_files()
    n_changed = 0
    for p in files:
        original = p.read_text()
        if p.suffix == ".py":
            new = strip_py_comments(original)
        else:
            new = strip_hash_comments_linewise(original)
        if new != original:
            n_changed += 1
            if dry_run:
                print(f"  WOULD CHANGE: {p.relative_to(ROOT)}")
            else:
                p.write_text(new)
                print(f"  stripped: {p.relative_to(ROOT)}")
    print(f"\n{'(dry run) ' if dry_run else ''}Changed {n_changed} / {len(files)} files")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
