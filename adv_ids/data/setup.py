"""Check raw-data layout and optionally fetch a public UNSW-NB15 training CSV."""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path

from adv_ids.data.catalog import (
    DatasetLayoutError,
    get_catalog,
    list_present_csvs,
    missing_data_message,
    require_dataset_files,
)

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(name: str, root: str | Path = ".") -> dict:
    """Inspect `data/raw/...` and return a status dict (does not download)."""
    entry = get_catalog(name)
    folder = Path(root) / entry.raw_dir
    present = list_present_csvs(name, root)
    status = {
        "dataset": entry.name,
        "raw_dir": str(folder),
        "exists": folder.exists(),
        "n_csv": len(present),
        "files": [str(p.relative_to(root)) if Path(root) in p.parents or p.is_relative_to(Path(root)) else str(p) for p in present],
        "homepage": entry.homepage,
        "ready": len(present) > 0,
        "hint": None if present else missing_data_message(name, root),
    }
    if present:
        status["sha256"] = {p.name: sha256_file(p) for p in present[:8]}
    return status


def _download(url: str, dest: Path, max_bytes: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "adversarial-gan-ids/research"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"{url} exceeded the {max_bytes} byte fetch cap.")
    if len(data) < 200:
        raise RuntimeError(f"{url} returned a tiny payload ({len(data)} bytes); refusing to save.")
    dest.write_bytes(data)
    return dest


def fetch_dataset(name: str, root: str | Path = ".", dest_name: str | None = None) -> Path:
    """Download the first working public URL for a dataset (currently UNSW training set)."""
    entry = get_catalog(name)
    if not entry.fetch_urls:
        raise DatasetLayoutError(
            f"No automatic fetch URLs for '{entry.name}'. {missing_data_message(name, root)}"
        )
    folder = Path(root) / entry.raw_dir
    target = folder / (dest_name or entry.example_files[0])
    errors: list[str] = []
    for url in entry.fetch_urls:
        try:
            _download(url, target, entry.max_fetch_bytes)
            digest = sha256_file(target)
            sums = folder / "SHA256SUMS"
            sums.write_text(f"{digest}  {target.name}\n", encoding="utf-8")
            logger.info("Saved %s (%d bytes, sha256=%s)", target, target.stat().st_size, digest)
            return target
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("Fetch failed: %s", errors[-1])
    raise DatasetLayoutError(
        "Automatic fetch failed for every mirror.\n  "
        + "\n  ".join(errors)
        + "\n\n"
        + missing_data_message(name, root)
    )


def setup_dataset(name: str, root: str | Path = ".", fetch: bool = False) -> dict:
    status = check_dataset(name, root)
    if status["ready"]:
        return status
    if fetch:
        path = fetch_dataset(name, root)
        status = check_dataset(name, root)
        status["fetched"] = str(path)
        return status
    raise DatasetLayoutError(status["hint"] or missing_data_message(name, root))


def print_status(status: dict) -> None:
    print(f"dataset:  {status['dataset']}")
    print(f"raw_dir:  {status['raw_dir']}")
    print(f"ready:    {status['ready']}  ({status['n_csv']} csv)")
    for f in status.get("files") or []:
        print(f"  - {f}")
    if status.get("sha256"):
        print("sha256 (first files):")
        for name, digest in status["sha256"].items():
            print(f"  {digest}  {name}")
    if status.get("hint"):
        print(status["hint"])
