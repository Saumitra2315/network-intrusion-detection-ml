"""Download the inspected CSV mirror and reject changed content."""

import argparse
import tempfile
import urllib.request
from pathlib import Path

from nids import DATA_FILES, MIRROR_BASE, sha256


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for filename, expected in DATA_FILES.items():
        target = directory / filename
        if target.exists():
            if sha256(target) != expected["sha256"]:
                raise ValueError(f"{target}: checksum mismatch; existing file was left unchanged.")
            print(f"Already present with expected SHA-256: {filename}")
            continue
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            partial = Path(staging) / filename
            with urllib.request.urlopen(MIRROR_BASE + filename, timeout=120) as response:
                with partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if sha256(partial) != expected["sha256"]:
                raise ValueError(f"{filename}: mirror content changed; checksum mismatch.")
            partial.rename(target)
        print(f"Downloaded with expected SHA-256: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        download(args.directory)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Download failed: {error}\n")
