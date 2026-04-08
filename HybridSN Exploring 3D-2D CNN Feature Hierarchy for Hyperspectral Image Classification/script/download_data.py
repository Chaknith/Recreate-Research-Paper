"""Download benchmark hyperspectral datasets used by HybridSN.

Datasets supported:
- Indian Pines (corrected) + ground truth
- Pavia University + ground truth
- Salinas (corrected) + ground truth

Default layout:
    data/
      IP/
        Indian_pines_corrected.mat
        Indian_pines_gt.mat
      UP/
        PaviaU.mat
        PaviaU_gt.mat
      SA/
        Salinas_corrected.mat
        Salinas_gt.mat

Examples:
    python scripts/download_data.py
    python scripts/download_data.py --datasets IP UP
    python scripts/download_data.py --root data --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CHUNK_SIZE = 1024 * 1024  # 1 MiB
DEFAULT_TIMEOUT = 60
USER_AGENT = "Mozilla/5.0 (compatible; HybridSN-dataset-downloader/1.0)"

DatasetFiles = Dict[str, Tuple[str, str]]

DATASETS: Dict[str, Dict[str, object]] = {
    "IP": {
        "folder": "IP",
        "description": "Indian Pines (corrected)",
        "files": {
            "Indian_pines_corrected.mat": (
                "https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
                "Corrected Indian Pines scene",
            ),
            "Indian_pines_gt.mat": (
                "https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
                "Indian Pines ground truth",
            ),
        },
    },
    "UP": {
        "folder": "UP",
        "description": "Pavia University",
        "files": {
            "PaviaU.mat": (
                "https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
                "Pavia University scene",
            ),
            "PaviaU_gt.mat": (
                "https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
                "Pavia University ground truth",
            ),
        },
    },
    "SA": {
        "folder": "SA",
        "description": "Salinas (corrected)",
        "files": {
            "Salinas_corrected.mat": (
                "https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
                "Corrected Salinas scene",
            ),
            "Salinas_gt.mat": (
                "https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
                "Salinas ground truth",
            ),
        },
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Indian Pines, Pavia University, and Salinas datasets from EHU."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS.keys()),
        default=sorted(DATASETS.keys()),
        help="Dataset codes to download. Default: all supported datasets.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../data"),
        help="Root directory where dataset folders will be created. Default: ../data",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the supported datasets and files, then exit.",
    )
    return parser


def print_catalog() -> None:
    print("Supported datasets:")
    for code in sorted(DATASETS.keys()):
        entry = DATASETS[code]
        print(f"  {code}: {entry['description']}")
        for filename, (_, label) in entry["files"].items():
            print(f"    - {filename}: {label}")


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def format_size(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


import re
import time
import urllib.request
from pathlib import Path

USER_AGENT = "Mozilla/5.0"
CHUNK_SIZE = 1024 * 1024  # 1 MiB


def format_size(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    return f"{mib:.1f} MiB"


def _parse_total_size(response) -> int | None:
    content_range = response.headers.get("Content-Range")
    if content_range:
        # Example: bytes 100-999/12345
        m = re.match(r"bytes\s+\d+-\d+/(\d+|\*)", content_range)
        if m and m.group(1).isdigit():
            return int(m.group(1))

    content_length = response.headers.get("Content-Length")
    if content_length and content_length.isdigit():
        return int(content_length)

    return None


def download_file(url: str, destination: Path, timeout: int, max_retries: int = 5) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    last_error = None
    total_bytes = None

    for attempt in range(1, max_retries + 1):
        try:
            existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0

            headers = {
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "identity",
            }

            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            request = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())

                # If server ignored Range and sent full file, restart from scratch.
                if existing_size > 0 and status != 206:
                    tmp_path.unlink(missing_ok=True)
                    existing_size = 0

                mode = "ab" if existing_size > 0 and status == 206 else "wb"

                response_total = _parse_total_size(response)

                # For 206 responses, _parse_total_size returns the full file size from Content-Range.
                # For 200 responses, Content-Length is only the current response body size.
                if status == 206:
                    total_bytes = response_total
                elif status == 200 and response_total is not None:
                    total_bytes = response_total

                downloaded = existing_size

                with tmp_path.open(mode) as out:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)

                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            print(
                                f"\r      {format_size(downloaded)} / {format_size(total_bytes)} ({pct:5.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(f"\r      {format_size(downloaded)}", end="", flush=True)

            print()

            final_size = tmp_path.stat().st_size

            if total_bytes is not None and final_size != total_bytes:
                raise IOError(
                    f"Incomplete download: got {final_size} bytes, expected {total_bytes} bytes"
                )

            tmp_path.replace(destination)
            return final_size

        except Exception as exc:
            last_error = exc
            print(f"\n    Attempt {attempt}/{max_retries} failed: {exc}")

            # Keep partial file for resume on next attempt.
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise last_error



def iter_selected_files(dataset_codes: Iterable[str]) -> Iterable[Tuple[str, str, Path, str, str]]:
    for code in dataset_codes:
        entry = DATASETS[code]
        folder = Path(str(entry["folder"]))
        description = str(entry["description"])
        files: DatasetFiles = entry["files"]  # type: ignore[assignment]
        for filename, (url, label) in files.items():
            yield code, description, folder / filename, url, label



def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        print_catalog()
        return 0

    selected = list(dict.fromkeys(args.datasets))
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading to: {root.resolve()}")
    print(f"Datasets: {', '.join(selected)}")

    failures: List[str] = []

    for code in selected:
        dataset_entry = DATASETS[code]
        dataset_dir = root / str(dataset_entry["folder"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{code}] {dataset_entry['description']}")

        files: DatasetFiles = dataset_entry["files"]  # type: ignore[assignment]
        for filename, (url, label) in files.items():
            destination = dataset_dir / filename
            print(f"  - {label}")
            print(f"    URL: {url}")
            print(f"    Save: {destination}")

            if destination.exists() and not args.overwrite:
                size = destination.stat().st_size
                digest = sha256_of_file(destination)
                print(f"    Status: already exists, skipping ({format_size(size)}, sha256={digest[:12]}...)")
                continue

            try:
                downloaded = download_file(url, destination, timeout=args.timeout)
                digest = sha256_of_file(destination)
                print(f"    Done: {format_size(downloaded)}, sha256={digest}")
            except urllib.error.HTTPError as exc:
                failures.append(f"{destination.name}: HTTP {exc.code} at {url}")
                print(f"    Error: HTTP {exc.code} while downloading {url}")
            except urllib.error.URLError as exc:
                failures.append(f"{destination.name}: URL error {exc.reason} at {url}")
                print(f"    Error: URL error while downloading {url}: {exc.reason}")
            except Exception as exc:  # pragma: no cover - defensive fallback
                failures.append(f"{destination.name}: {exc}")
                print(f"    Error: {exc}")

    if failures:
        print("\nSome downloads failed:")
        for item in failures:
            print(f"  - {item}")
        print("\nTip: if EHU changes a link, open the source page and update the URL in DATASETS.")
        return 1

    print("\nAll requested files are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
