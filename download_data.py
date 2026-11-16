"""Download data/nanotabpfn_model_tabiclv1_prior.csv from figshare and check its MD5.

Usage: python download_data.py [--force]
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

DOI = "https://doi.org/10.6084/m9.figshare.34912776"
URL = "https://ndownloader.figshare.com/files/71080338"
MD5 = "ddcdee8658c25a2a3286179a045ce31f"
TARGET = Path(__file__).resolve().parent / "data" / "nanotabpfn_model_tabiclv1_prior.csv"
CHUNK = 1 << 20


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if TARGET.exists() and "--force" not in sys.argv:
        if md5(TARGET) == MD5:
            print(f"{TARGET} already present and verified.")
            return
        print(f"{TARGET} exists but its MD5 does not match; downloading again.")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ({DOI}) ...")
    with urllib.request.urlopen(URL) as response, TARGET.open("wb") as out:  # noqa: S310 (fixed https URL)
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        for chunk in iter(lambda: response.read(CHUNK), b""):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    actual = md5(TARGET)
    if actual != MD5:
        TARGET.unlink()
        raise SystemExit(f"MD5 mismatch ({actual} != {MD5}); the file was removed, please retry.")
    print(f"Saved and verified {TARGET}.")


if __name__ == "__main__":
    main()
