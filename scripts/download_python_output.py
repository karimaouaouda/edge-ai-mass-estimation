"""Download matching files from a Kaggle kernel output, including all pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="aouaoudakarim/yolo-seg")
    parser.add_argument("--pattern", default=r"\.onnx$")
    parser.add_argument("--output-dir", type=Path, default=Path("kaggle_outputs"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()

    # Kaggle output listings are paginated. Pass each returned token back to
    # the API so large training runs do not silently omit later artifacts.
    page_token: str | None = None
    downloaded: list[str] = []
    while True:
        files, page_token = api.kernels_output(
            args.kernel,
            path=str(args.output_dir),
            file_pattern=args.pattern,
            force=args.force,
            quiet=False,
            page_token=page_token,
        )
        downloaded.extend(files)
        if not page_token:
            break

    print(f"Downloaded {len(downloaded)} file(s) to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
