"""Check raw-data layout and optionally fetch public UNSW / Engelen dumps."""

from __future__ import annotations

import hashlib
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from adv_ids.data.catalog import (
    DatasetLayoutError,
    get_catalog,
    list_present_csvs,
    missing_data_message,
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
        "files": [
            str(p.relative_to(root)) if Path(root) in p.parents or p.is_relative_to(Path(root)) else str(p)
            for p in present
        ],
        "homepage": entry.homepage,
        "ready": len(present) > 0,
        "hint": None if present else missing_data_message(name, root),
    }
    if present:
        status["sha256"] = {p.name: sha256_file(p) for p in present[:8]}
    return status


def _looks_like_html(header: bytes, content_type: str | None) -> bool:
    if content_type and "html" in content_type.lower():
        return True
    start = header.lstrip().lower()
    return start.startswith(b"<!doctype html") or start.startswith(b"<html")


def _download(url: str, dest: Path, max_bytes: int) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "adversarial-gan-ids/research"})
    written = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        content_type = resp.headers.get("Content-Type")
        first = resp.read(2048)
        if _looks_like_html(first, content_type):
            raise RuntimeError(
                f"{url} returned HTML ({content_type!r}), not a data file. "
                "This is usually a registration / login / landing page."
            )
        with tmp.open("wb") as fh:
            fh.write(first)
            written += len(first)
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"{url} exceeded the {max_bytes} byte fetch cap.")
                fh.write(chunk)
    if written < 200:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} returned a tiny payload ({written} bytes); refusing to save.")
    tmp.replace(dest)
    return dest


def _append_sha256(folder: Path, dest: Path) -> str:
    digest = sha256_file(dest)
    sums = folder / "SHA256SUMS"
    line = f"{digest}  {dest.name}\n"
    existing = sums.read_text(encoding="utf-8") if sums.is_file() else ""
    others = [ln for ln in existing.splitlines() if not ln.endswith(f"  {dest.name}")]
    sums.write_text("\n".join(others + [line.strip()]) + "\n", encoding="utf-8")
    return digest


def fetch_url_to(url: str, dest: Path, max_bytes: int) -> Path:
    folder = dest.parent
    _download(url, dest, max_bytes)
    digest = _append_sha256(folder, dest)
    logger.info("Saved %s (%d bytes, sha256=%s)", dest, dest.stat().st_size, digest)
    return dest


def fetch_dataset(name: str, root: str | Path = ".", dest_name: str | None = None) -> Path:
    """Download the first working public URL for a dataset file."""
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
            return fetch_url_to(url, target, entry.max_fetch_bytes)
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            errors.append(f"{url}: {exc}")
            logger.warning("Fetch failed: %s", errors[-1])
    raise DatasetLayoutError(
        "Automatic fetch failed for every mirror.\n  "
        + "\n  ".join(errors)
        + "\n\n"
        + missing_data_message(name, root)
    )


def fetch_named_files(name: str, root: str | Path = ".", overwrite: bool = False) -> list[Path]:
    """Fetch every catalog `fetch_files` target that is missing (or overwrite)."""
    entry = get_catalog(name)
    folder = Path(root) / entry.raw_dir
    saved: list[Path] = []
    errors: list[str] = []
    for dest_name, urls in entry.fetch_files or ():
        dest = folder / dest_name
        if dest.is_file() and not overwrite and dest.stat().st_size > 1000:
            logger.info("Already present: %s", dest)
            saved.append(dest)
            continue
        last_err = None
        for url in urls:
            try:
                saved.append(fetch_url_to(url, dest, entry.max_fetch_bytes))
                last_err = None
                break
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                last_err = f"{url}: {exc}"
                logger.warning("Fetch failed: %s", last_err)
        if last_err:
            errors.append(f"{dest_name}: {last_err}")
    if entry.extract_zip and saved:
        _maybe_unzip(folder, entry.extract_zip)
    if not saved:
        raise DatasetLayoutError(
            "Automatic fetch failed.\n  " + "\n  ".join(errors) + "\n\n" + missing_data_message(name, root)
        )
    return saved


def _maybe_unzip(folder: Path, zip_name: str) -> None:
    zpath = folder / zip_name
    if not zpath.is_file():
        return
    with zipfile.ZipFile(zpath) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csvs:
            dest = folder / Path(name).name
            if dest.is_file() and dest.stat().st_size > 1000:
                continue
            logger.info("Extracting %s → %s", name, dest)
            dest.write_bytes(zf.read(name))


def setup_dataset(name: str, root: str | Path = ".", fetch: bool = False) -> dict:
    status = check_dataset(name, root)
    if fetch:
        entry = get_catalog(name)
        if entry.fetch_files:
            fetch_named_files(name, root)
        elif not status["ready"]:
            fetch_dataset(name, root)
        elif entry.fetch_urls:
            fetch_dataset(name, root)
        status = check_dataset(name, root)
        return status
    if not status["ready"]:
        raise DatasetLayoutError(status["hint"] or missing_data_message(name, root))
    return status


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
