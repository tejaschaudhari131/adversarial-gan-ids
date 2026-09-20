#!/usr/bin/env python3
"""Print layout / checksum notes and optionally fetch a public UNSW training CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adv_ids.data.catalog import CATALOG, DatasetLayoutError, missing_data_message
from adv_ids.data.setup import check_dataset, print_status, setup_dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check or fetch IDS flow-feature CSVs into data/raw/.")
    p.add_argument("--dataset-name", default="unsw_nb15", choices=sorted(CATALOG))
    p.add_argument("--fetch", action="store_true", help="Try public mirrors (UNSW training set only).")
    p.add_argument("--root", default=str(ROOT))
    args = p.parse_args(argv)
    try:
        if args.fetch:
            status = setup_dataset(args.dataset_name, root=args.root, fetch=True)
        else:
            status = check_dataset(args.dataset_name, root=args.root)
            if not status["ready"]:
                print(missing_data_message(args.dataset_name, args.root))
                return 2
        print_status(status)
        return 0
    except DatasetLayoutError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
