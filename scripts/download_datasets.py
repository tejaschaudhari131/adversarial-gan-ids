#!/usr/bin/env python3
"""Fetch public UNSW-NB15 / Engelen CIC-IDS2017 dumps into data/raw/ (gitignored).

Official CIC MachineLearningCSV.zip hosts currently return an HTML portal
(HTTP 200, ~109 KB), not the zip — this script records that and does not
pretend the download succeeded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adv_ids.data.catalog import CATALOG, DatasetLayoutError
from adv_ids.data.setup import check_dataset, fetch_named_files, print_status, setup_dataset


def _probe_official_cic() -> None:
    import urllib.request

    urls = [
        "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
        "https://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip",
    ]
    print("Official CIC MachineLearningCSV.zip probes (expect HTML portal):")
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "adversarial-gan-ids/research"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                chunk = resp.read(64)
                ctype = resp.headers.get("Content-Type")
                print(f"  {resp.status} {ctype} {url}  head={chunk[:40]!r}")
        except Exception as exc:
            print(f"  FAIL {url}: {exc}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-name",
        action="append",
        dest="datasets",
        help="unsw_nb15 and/or cicids2017. Default: both.",
    )
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--probe-cic-portal", action="store_true", help="HEAD/GET official CIC zip URLs.")
    args = p.parse_args(argv)
    names = args.datasets or ["unsw_nb15", "cicids2017"]
    if args.probe_cic_portal:
        _probe_official_cic()
    rc = 0
    for name in names:
        if name not in CATALOG:
            print(f"Unknown dataset {name}", file=sys.stderr)
            rc = 2
            continue
        try:
            if CATALOG[name].fetch_files:
                fetch_named_files(name, root=args.root)
            else:
                setup_dataset(name, root=args.root, fetch=True)
            print_status(check_dataset(name, root=args.root))
        except DatasetLayoutError as exc:
            print(exc, file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
