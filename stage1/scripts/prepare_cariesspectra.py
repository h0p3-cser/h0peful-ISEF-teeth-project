"""Prepare a portable split of Caries-Spectra v2 (DOI 10.17632/9jnf2jvghy.2)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["early", "advanced", "no_caries"]
CLASS_FOLDERS = dict(
    early="EarlyStageEnamel_Caries",
    advanced="AdvanceEnamel_Caries",
    no_caries="NoEnamel_Caries",
)
FOLDER_HINTS = {
    "early": ("early",),
    "advanced": ("advance",),
    "no_caries": ("noenamel", "nocaries", "normal", "healthy"),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def extract_archive(archive: Path, target: Path) -> None:
    target.mkdir()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            member = PurePosixPath(entry.filename)
            if (
                member.is_absolute()
                or PureWindowsPath(entry.filename).drive
                or "\\" in entry.filename
                or ".." in member.parts
                or any(":" in part for part in member.parts)
                or stat.S_ISLNK(entry.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
        source.extractall(target)


def collect(source: Path) -> dict[str, list[Path]]:
    directories = sorted(source.rglob("*"))
    result = {}
    for label in CLASSES:
        exact = [p for p in directories if p.is_dir() and p.name.casefold() == CLASS_FOLDERS[label].casefold()]
        selected = exact or [
            path for path in directories
            if path.is_dir()
            and not any(part.startswith("[") or "original" in part.lower() for part in path.relative_to(source).parts)
            and any(hint in path.name.lower().replace(" ", "").replace("-", "").replace("_", "")
                    for hint in FOLDER_HINTS[label])
            and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in path.iterdir())
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one source folder for {label}, found {len(selected)}")
        result[label] = sorted(path for path in selected[0].iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
    return result


def materialize(source: Path, target: Path, val: float, test: float, seed: int) -> Counter:
    found = collect(source)
    rng = random.Random(seed)
    seen = set()
    rows = []
    counts = Counter()
    for label, images in found.items():
        n_val = round(len(images) * val)
        n_test = round(len(images) * test)
        n_train = len(images) - n_val - n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(f"At least one {label} image per split is required")
        rng.shuffle(images)
        parts = {"train": images[:n_train], "val": images[n_train:n_train + n_val], "test": images[n_train + n_val:]}
        destination = target / "images" / label
        destination.mkdir(parents=True)
        for split, group in parts.items():
            for photo in group:
                content = photo.read_bytes()
                digest = hashlib.sha256(content).digest()
                if digest in seen:
                    raise ValueError(f"Identical source image found twice: {photo.name}")
                seen.add(digest)
                copy = destination / photo.name
                copy.write_bytes(content)
                rows.append({"image_path": copy.relative_to(target).as_posix(), "class": label, "split": split})
            counts[(split, label)] = len(group)
    with (target / "manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Official ZIP or extracted dataset directory")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "cariesspectra", help="New prepared dataset directory")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not (0 < args.val < 1 and 0 < args.test < 1 and args.val + args.test < 1):
        parser.error("Use positive validation and test fractions with sum below one")
    source = args.src.resolve()
    output = args.out.resolve()
    is_archive = source.is_file() and source.suffix.lower() == ".zip"
    if not is_archive and not source.is_dir():
        parser.error(f"Need an existing ZIP or directory: {source}")
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    if source.is_dir() and output.is_relative_to(source):
        parser.error("Put the prepared dataset outside the source directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".cariesspectra-", dir=output.parent) as temp:
            workspace = Path(temp).resolve()
            if is_archive:
                extracted = workspace / "raw"
                extract_archive(source, extracted)
                source = extracted
            # Mendeley Download All wraps the original archive in another ZIP.
            inner_archives = sorted(source.rglob("Caries_Dataset.zip"))
            if len(inner_archives) > 1:
                raise ValueError("Choose one Caries_Dataset.zip as the source")
            if inner_archives:
                extracted = workspace / "dataset"
                extract_archive(inner_archives[0], extracted)
                source = extracted
            prepared = workspace / "prepared"
            counts = materialize(source, prepared, args.val, args.test, args.seed)
            if output.exists():
                raise FileExistsError(f"Output directory already exists: {output}")
            prepared.rename(output)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Dataset preparation failed: {error}\n")
    print(f"Manifest: {output / 'manifest.csv'} ({sum(counts.values())} images)")
    for split in ("train", "val", "test"):
        print(split + ": " + ", ".join(f"{label}={counts[(split, label)]}" for label in CLASSES))


if __name__ == "__main__":
    main()
