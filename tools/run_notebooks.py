"""
Execute notebooks from the command line and fail if any cell raises an error.

Usage (from the repository root)::

    python tools/run_notebooks.py                      # all notebooks in PrestazioniRoskam/
    python tools/run_notebooks.py path/to/a.ipynb ...  # selected notebooks
    python tools/run_notebooks.py --inplace            # also save the executed outputs

Used by the continuous-integration workflow to guarantee that the notebooks
stay consistent with the ``aircraft_performance`` package.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "PrestazioniRoskam"


def run(path: pathlib.Path, inplace: bool, timeout: int) -> bool:
    """Execute one notebook; return True on success."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as exc:
        print(f"FAILED  {path.relative_to(ROOT)}\n{exc}", file=sys.stderr)
        return False
    if inplace:
        nbformat.write(nb, path)
    print(f"OK      {path.relative_to(ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=pathlib.Path)
    parser.add_argument("--inplace", action="store_true", help="save the executed notebooks")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout [s]")
    args = parser.parse_args()
    paths = [p.resolve() for p in args.notebooks] or sorted(DEFAULT_DIR.glob("*.ipynb"))
    if not paths:
        print("No notebooks found.", file=sys.stderr)
        return 1
    results = [run(p, args.inplace, args.timeout) for p in paths]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
