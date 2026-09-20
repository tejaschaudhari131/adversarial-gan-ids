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
    fetch_files: tuple[tuple[str, tuple[str, ...]], ...] = ()
    extract_zip: str | None = None
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
            "Official MachineLearningCSV.zip URLs (205.174.165.80 / cicresearch.ca) "
            "return an HTML registration portal (HTTP 200, ~109 KB HTML), not the zip. "
            "A public no-login alternative is the Engelen-corrected regeneration: "
            "https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip "
            "(~319 MB). `python run.py setup-data --dataset-name cicids2017 --fetch` "
            "tries that zip and extracts day CSVs into data/raw/cicids2017/."
        ),
        checksum_notes=(
            "Observed 2026-09-20 for Engelen dataset.zip: "
            "sha256=4d535da19795d85376ae1397d161329e3b06fc47d9a5a68cd9be2cd7ecee0f2a "
            "(333,841,436 bytes). Official CIC does not publish a stable SHA-256."
        ),
        size_note="Engelen zip is ~319 MB; uncompressed day CSVs are ~1.1 GB. Do not commit them.",
        fetch_files=(
            (
                "engelen_WTMC2021_dataset.zip",
                (
                    "https://intrusion-detection.distrinet-research.be/WTMC2021/Dataset/dataset.zip",
                    "https://downloads.distrinet-research.be/WTMC2021/Dataset/dataset.zip",
                ),
            ),
        ),
        extract_zip="engelen_WTMC2021_dataset.zip",
        max_fetch_bytes=400_000_000,
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
            "`python run.py setup-data --dataset-name unsw_nb15 --fetch` pulls the "
            "Hugging Face copy of the 175k training CSV and a public 82k testing CSV."
        ),
        checksum_notes=(
            "Observed 2026-09-20: training "
            "sha256=bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa "
            "(32,293,018 bytes, 175,341 rows, Hugging Face Mouwiya/UNSW-NB15). "
            "testing sha256=7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c "
            "(15,298,467 bytes, 82,332 rows, ushukkla/nospammers; published test size)."
        ),
        size_note="Official ML-export CSVs are tens of MB (not multi-GB). Still gitignored.",
        fetch_urls=(
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
            "https://github.com/ushukkla/nospammers/raw/master/UNSW_NB15_training-set.csv",
        ),
        fetch_files=(
            (
                "UNSW_NB15_training-set.csv",
                (
                    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
                ),
            ),
            (
                "UNSW_NB15_testing-set.csv",
                (
                    "https://github.com/ushukkla/nospammers/raw/master/UNSW_NB15_training-set.csv",
                ),
            ),
        ),
        max_fetch_bytes=80_000_000,
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
