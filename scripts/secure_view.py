"""Show one tier-3 validation item in the terminal, for human labelling only (2026-10-22).

Reads the private validation key and the local, never-published Qwen3.6 run file. Prints
the prompt and completion for a single sheet id. Writes nothing, makes no network calls,
and refuses ids that are not tier-3 VIEW-LOCALLY rows.

Usage (on darkstar, from the repo root):
    python scripts/secure_view.py <sheet_id> [--key <path to validation_key.json>]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap
from pathlib import Path

DEFAULT_KEY = Path(
    "/home/darkstar/chem-bench-qwen36-2026-08-12/results/biorefusalaudit/private/"
    "validation_2026-10/validation_key.json"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet_id")
    ap.add_argument("--key", default=str(DEFAULT_KEY))
    args = ap.parse_args()
    key = json.loads(Path(args.key).read_text())
    entry = key.get(args.sheet_id)
    if not entry or entry.get("source") != "qwen36_local":
        print("not a VIEW-LOCALLY row", file=sys.stderr)
        return 2
    data = json.loads(Path(entry["qwen36_file"]).read_text())
    rec = next(r for r in data["records"] if r["id"] == entry["record_id"])
    width = min(100, shutil.get_terminal_size().columns)
    print(f"[{args.sheet_id}] tier-3 item. Do not copy this text anywhere.\n")
    print("PROMPT:")
    print(textwrap.fill(rec["prompt"], width))
    print("\nCOMPLETION:")
    for para in (rec.get("completion") or "").split("\n"):
        print(textwrap.fill(para, width) if para.strip() else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
