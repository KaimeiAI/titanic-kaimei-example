"""Run the example notebooks top to bottom and save their outputs in place.

    python scripts/run_notebooks.py            # 01 to 04 (needs KAIMEI_API_KEY or a .env file)
    python scripts/run_notebooks.py 00 02      # only these
    python scripts/run_notebooks.py --check    # run into a temporary folder; leave the notebooks alone

Notebook 00 (the original Kaggle notebook, no Kaimei calls) is included when named or with --all.
The notebooks depend on each other: 02, 03 and 04 use the models that 01 uploads.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("only", nargs="*", help="notebook number prefixes to run, e.g. 01 03")
    parser.add_argument("--all", action="store_true", help="include notebook 00")
    parser.add_argument("--check", action="store_true", help="don't overwrite the notebooks")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    args = parser.parse_args()

    paths = sorted(NOTEBOOKS.glob("[0-9][0-9]-*.ipynb"))
    if args.only:
        paths = [p for p in paths if any(p.name.startswith(prefix) for prefix in args.only)]
    elif not args.all:
        paths = [p for p in paths if not p.name.startswith("00")]
    if not paths:
        print("No notebooks matched.", file=sys.stderr)
        return 2

    scratch = Path(tempfile.mkdtemp(prefix="titanic-notebooks-")) if args.check else None
    failed = 0
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        started = time.time()
        client = NotebookClient(notebook, timeout=args.timeout, kernel_name="python3",
                                resources={"metadata": {"path": str(NOTEBOOKS)}})
        try:
            client.execute()
            status = "ok"
        except CellExecutionError as exc:
            failed += 1
            status = "FAILED\n" + str(exc).splitlines()[-1]
        target = scratch / path.name if scratch else path
        nbformat.write(notebook, target)
        print(f"{path.name}: {status} ({time.time() - started:.0f} s) -> {target}")
    if scratch and not failed:
        shutil.rmtree(scratch)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
