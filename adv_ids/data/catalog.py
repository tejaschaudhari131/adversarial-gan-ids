"""Official dataset layout, download notes, and actionable missing-file errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetCatalogEntry:
    name: str
    raw_dir: str
    expected_globs: tuple[str, ...]
    example_files: tuple[str, ...]
    homepage: str
    download_notes: str
    checksum_notes: str
    size_note: str
    fetch_urls: tuple[str, ...] = ()
    max_fetch_bytes: int = 80_000_000


CATALOG: dict[str, DatasetCatalogEntry] = {
    "cicids2017": DatasetCatalogEntry(
        name="cicids2017",
        raw_dir="data/raw/cicids2017",
        expected_globs=("*.csv",),
        example_files=(
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Tuesday-WorkingHours.pcap_ISCX.csv",
        ),
        homepage="https://www.unb.ca/cic/datasets/ids-2017.html",
        download_notes=(
            "Download MachineLearningCSV.zip from the CIC IDS 2017 page, unzip, "
            "and place the day CSVs in data/raw/cicids2017/. Prefer the Engelen "
            "et al. reconstructed features when writing a paper "
            "(https://intrusion-detection.distrinet-research.be/WTMC2021/)."
        ),
        checksum_notes=(
            "CIC does not publish a stable SHA-256 for MachineLearningCSV.zip. "
            "After download, record `sha256sum MachineLearningCSV.zip` in "
            "docs/DATA_CARD.md (you write this file locally; do not commit the zip)."
        ),
        size_note="MachineLearningCSV.zip is hundreds of MB. Do not commit it.",
    ),
    "cicids2018": DatasetCatalogEntry(
        name="cicids2018",
        raw_dir="data/raw/cse-cic-ids2018",
        expected_globs=("*.csv",),
        example_files=(
            "Wednesday-14-02-2018.csv",
            "Thursday-15-02-2018.csv",
            "Friday-16-02-2018.csv",
        ),
        homepage="https://www.unb.ca/cic/datasets/ids-2018.html",
        download_notes=(
            "CSE-CIC-IDS2018 CSVs are distributed via AWS Open Data: "
            "https://registry.opendata.aws/cse-cic-ids2018/ . Copy a 2–3 day "
            "subset into data/raw/cse-cic-ids2018/ (see IDS2018_RECOMMENDED_DAYS)."
        ),
        checksum_notes=(
            "Hash each day CSV you keep (`sha256sum data/raw/cse-cic-ids2018/*.csv`) "
            "and paste the digests into your local docs/DATA_CARD.md."
        ),
        size_note="Per-day CSVs are often 1–10+ GB. Do not commit them.",
    ),
    "unsw_nb15": DatasetCatalogEntry(
        name="unsw_nb15",
        raw_dir="data/raw/unsw-nb15",
        expected_globs=("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv", "*.csv"),
        example_files=("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"),
        homepage="https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        download_notes=(
            "Prefer the official ML export: UNSW_NB15_training-set.csv (175,341 rows) "
            "and UNSW_NB15_testing-set.csv (82,332 rows) from the UNSW project page. "
            "`python run.py setup-data --dataset-name unsw_nb15 --fetch` tries a short "
            "list of public mirrors of that export; if they fail, download manually. "
            "One working mirror (ushukkla/nospammers) served a file *named* "
            "UNSW_NB15_training-set.csv whose row count matched the published "
            "testing set (82,332). Always check the row count after fetch."
        ),
        checksum_notes=(
            "No single vendor SHA-256 is guaranteed across mirrors. After a fetch, "
            "this repo writes data/raw/unsw-nb15/SHA256SUMS for the files it saved. "
            "Observed digest for the ushukkla/nospammers copy (82,332 rows): "
            "7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c. "
            "Record whatever digest you actually downloaded in docs/DATA_CARD.md."
        ),
        size_note="The official ML-export CSVs are tens of MB (not multi-GB). Still gitignored.",
        fetch_urls=(
            # Public copies of the official ML export (not the 2M-row packet dump).
            "https://github.com/ushukkla/nospammers/raw/master/UNSW_NB15_training-set.csv",
            "https://raw.githubusercontent.com/Nir-Az/UNSW-NB15-Dataset/main/UNSW_NB15_training-set.csv",
        ),
        max_fetch_bytes=60_000_000,
    ),
    "ciciot2023": DatasetCatalogEntry(
        name="ciciot2023",
        raw_dir="data/raw/cic-iot-2023",
        expected_globs=("*.csv",),
        example_files=("part-00000-*.csv",),
        homepage="https://www.unb.ca/cic/datasets/iotdataset-2023.html",
        download_notes="Optional stretch. Place sampled CIC-IoT-2023 CSVs under data/raw/cic-iot-2023/.",
        checksum_notes="Record sha256 of any part files you keep.",
        size_note="Full dump is multi-GB. Use a day sample only.",
    ),
}


def get_catalog(name: str) -> DatasetCatalogEntry:
    from adv_ids.data.schemas import normalize_dataset_name

    key = normalize_dataset_name(name)
    if key not in CATALOG:
        known = ", ".join(sorted(CATALOG))
        raise ValueError(f"Unknown dataset '{name}'. Known: {known}")
    return CATALOG[key]


def expected_raw_dir(name: str, root: str | Path = ".") -> Path:
    return Path(root) / get_catalog(name).raw_dir


def list_present_csvs(name: str, root: str | Path = ".") -> list[Path]:
    folder = expected_raw_dir(name, root)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*.csv") if p.is_file())


def missing_data_message(name: str, root: str | Path = ".", extra_path: str | Path | None = None) -> str:
    entry = get_catalog(name)
    folder = expected_raw_dir(name, root)
    tried = extra_path or folder
    examples = "\n".join(f"    {folder / f}" for f in entry.example_files)
    return (
        f"Dataset '{entry.name}' is not on disk.\n"
        f"  Looked for: {tried}\n"
        f"  Expected directory: {folder}/\n"
        f"  Example files:\n{examples}\n"
        f"  Homepage: {entry.homepage}\n"
        f"  How to get the files: {entry.download_notes}\n"
        f"  Checksums: {entry.checksum_notes}\n"
        f"  Size: {entry.size_note}\n"
        f"  Next steps:\n"
        f"    python run.py setup-data --dataset-name {entry.name}\n"
        f"    python run.py setup-data --dataset-name {entry.name} --fetch   # UNSW only, optional\n"
        f"    python run.py prepare --dataset-name {entry.name} --synthetic\n"
        f"  Schema fixtures (tiny, committed): tests/fixtures/{entry.name}_official_sample.csv"
    )


class DatasetLayoutError(FileNotFoundError):
    """Raised when a real dataset path is required but missing."""


def require_dataset_files(
    name: str,
    path: str | Path | None = None,
    *,
    root: str | Path = ".",
) -> Path:
    """Return an existing CSV file or directory, or raise DatasetLayoutError."""
    if path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise DatasetLayoutError(missing_data_message(name, root, extra_path=candidate))
    folder = expected_raw_dir(name, root)
    present = list_present_csvs(name, root)
    if present:
        return folder
    raise DatasetLayoutError(missing_data_message(name, root, extra_path=folder))
