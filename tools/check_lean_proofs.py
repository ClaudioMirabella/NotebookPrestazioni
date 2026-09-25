"""
Static check of the Lean proofs in ``lean/``: no proof may be left unfinished.

Lean accepts a file containing ``sorry`` (with a warning only), so a successful
``lake build`` alone does not guarantee that every theorem is proved. This script
fails if, outside comments, any Lean file contains

* ``sorry`` / ``admit``          - an unfinished proof;
* ``axiom``                      - a new unproved assumption;
* ``native_decide``              - trusts the compiler instead of the kernel;
* ``implemented_by`` / ``extern``- replaces a definition by unchecked code.

Usage (from the repository root)::

    python tools/check_lean_proofs.py

It also lists the theorems and lemmas found, file by file. Exit code 0 = clean.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEAN_DIR = ROOT / "lean"
FORBIDDEN = ("sorry", "admit", "axiom", "native_decide", "implemented_by", "extern")


def strip_comments(source: str) -> str:
    """Remove Lean comments: nested block comments /- ... -/ and line comments -- ..."""
    out, depth, i = [], 0, 0
    while i < len(source):
        two = source[i:i + 2]
        if two == "/-":
            depth += 1
            i += 2
        elif two == "-/" and depth > 0:
            depth -= 1
            i += 2
        elif depth == 0 and two == "--":
            while i < len(source) and source[i] != "\n":
                i += 1
        else:
            if depth == 0:
                out.append(source[i])
            i += 1
    if depth != 0:
        raise ValueError("unbalanced block comment")
    return "".join(out)


def check_file(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return (problems, declaration names) for one Lean file."""
    code = strip_comments(path.read_text(encoding="utf-8"))
    problems = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        for word in FORBIDDEN:
            if re.search(rf"\b{word}\b", line):
                problems.append(f"{path.relative_to(ROOT)}:{lineno}: forbidden '{word}'")
    names = re.findall(r"^\s*(?:theorem|lemma)\s+([\w'.₀-₉]+)", code, flags=re.M)
    return problems, names


def main() -> int:
    files = sorted(LEAN_DIR.glob("AircraftPerformance/*.lean"))
    if not files:
        print(f"No Lean files found in {LEAN_DIR}", file=sys.stderr)
        return 1
    all_problems, total = [], 0
    for path in files:
        problems, names = check_file(path)
        all_problems += problems
        total += len(names)
        print(f"{path.name:22s} {len(names):3d} theorems/lemmas: {', '.join(names)}")
    if all_problems:
        print("\n".join(all_problems), file=sys.stderr)
        return 1
    print(f"OK: {total} theorems/lemmas, no sorry/admit/axiom in {len(files)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
