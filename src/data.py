"""Loading utilities for the Turkish (tr-TR) portion of the Amazon MASSIVE dataset.

MASSIVE is distributed by Amazon Science as a single gzipped tarball holding one
JSONL file per locale. We read that archive directly instead of going through the
Hugging Face `AmazonScience/massive` dataset, because that repository is a
script-based dataset and `datasets>=3.0` refuses to execute dataset scripts.

Reference:
    FitzGerald et al. (2022), "MASSIVE: A 1M-Example Multilingual Natural Language
    Understanding Dataset with 51 Typologically-Diverse Languages".
    https://arxiv.org/abs/2204.08582
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.0.tar.gz"
)
MASSIVE_SHA256 = "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
DEFAULT_LOCALE = "tr-TR"
DEFAULT_CACHE_DIR = Path("data/raw")

# MASSIVE calls the validation partition "dev"; we expose the HF-style name.
PARTITION_TO_SPLIT = {"train": "train", "dev": "validation", "test": "test"}
SPLIT_ORDER = ("train", "validation", "test")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    *,
    verify: bool = True,
) -> Path:
    """Download the MASSIVE tarball into `cache_dir`, skipping if already present."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "amazon-massive-dataset-1.0.tar.gz"

    if not archive.exists():
        print(f"Downloading MASSIVE (~38 MB) to {archive} ...")
        urllib.request.urlretrieve(MASSIVE_URL, archive)

    if verify:
        checksum = _sha256(archive)
        if checksum != MASSIVE_SHA256:
            raise RuntimeError(
                f"Checksum mismatch for {archive}.\n"
                f"  expected: {MASSIVE_SHA256}\n"
                f"  actual:   {checksum}\n"
                "Delete the file and retry the download."
            )
    return archive


def extract_locale(
    locale: str = DEFAULT_LOCALE,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> Path:
    """Extract a single locale's JSONL file from the tarball and return its path."""
    cache_dir = Path(cache_dir)
    jsonl_path = cache_dir / f"{locale}.jsonl"
    if jsonl_path.exists():
        return jsonl_path

    archive = download_archive(cache_dir)
    member_name = f"1.0/data/{locale}.jsonl"
    with tarfile.open(archive, "r:gz") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError as exc:
            raise KeyError(f"Locale {locale!r} not found in the MASSIVE archive.") from exc
        source = tar.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not read {member_name} from the archive.")
        jsonl_path.write_bytes(source.read())
    return jsonl_path


def _mean_judgment(judgments: Iterable[dict], field: str) -> float | None:
    """Average one annotator score across the (usually three) judgments of a row."""
    scores = [j[field] for j in judgments if j.get(field) is not None]
    return sum(scores) / len(scores) if scores else None


def load_locale(
    locale: str = DEFAULT_LOCALE,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Load every partition of one locale into a single tidy DataFrame.

    Columns: id, split, scenario, intent, utt, annot_utt, worker_id,
    plus the mean annotator scores intent_score / grammar_score / spelling_score.
    Annotator scores come from the dataset's own quality judgments and are used in
    the error analysis (low spelling scores flag orthography noise, which matters
    for Turkish diacritics).
    """
    jsonl_path = extract_locale(locale, cache_dir)

    records = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            judgments = row.get("judgments") or []
            records.append(
                {
                    "id": row["id"],
                    "split": PARTITION_TO_SPLIT[row["partition"]],
                    "scenario": row["scenario"],
                    "intent": row["intent"],
                    "utt": row["utt"],
                    "annot_utt": row.get("annot_utt"),
                    "worker_id": row.get("worker_id"),
                    "n_judgments": len(judgments),
                    "intent_score": _mean_judgment(judgments, "intent_score"),
                    "grammar_score": _mean_judgment(judgments, "grammar_score"),
                    "spelling_score": _mean_judgment(judgments, "spelling_score"),
                }
            )

    frame = pd.DataFrame.from_records(records)
    frame["split"] = pd.Categorical(frame["split"], categories=SPLIT_ORDER, ordered=True)
    return frame


def load_splits(
    locale: str = DEFAULT_LOCALE,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> dict[str, pd.DataFrame]:
    """Return {"train": df, "validation": df, "test": df} using the official splits."""
    frame = load_locale(locale, cache_dir)
    return {
        split: frame[frame["split"] == split].reset_index(drop=True)
        for split in SPLIT_ORDER
    }


def intent_labels(frame: pd.DataFrame) -> list[str]:
    """Sorted list of intent names — the canonical label order used everywhere.

    Deriving the order from sorted unique values keeps the label<->index mapping
    identical across the BERTurk classifier, the prompts and the metrics.
    """
    return sorted(frame["intent"].unique().tolist())


def label_mappings(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Build the label<->id maps used by the classifier heads and the results files."""
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


if __name__ == "__main__":
    splits = load_splits()
    for name, part in splits.items():
        print(f"{name:<11} {len(part):>6} rows")
    labels = intent_labels(pd.concat(splits.values()))
    print(f"intents     {len(labels):>6}")
