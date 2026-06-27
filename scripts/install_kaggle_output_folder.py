"""Install a folder from a Kaggle notebook output using the Kaggle API.

Example:
    python scripts/install_kaggle_output_folder.py \
        --kernel aouaoudakarim/yolo-seg \
        --output-folder data/normalized/realwaste \
        --install-dir data/normalized/realwaste \
        --force

Authentication uses the standard Kaggle API config:
    - KAGGLE_USERNAME and KAGGLE_KEY environment variables, or
    - ~/.kaggle/kaggle.json
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

def configure_utf8_streams() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "encoding", None) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def normalize_output_folder(folder: str) -> str:
    normalized = folder.replace("\\", "/").strip("/")
    if not normalized or normalized == ".":
        return ""
    if ".." in Path(normalized).parts:
        raise ValueError("--output-folder must not contain '..'")
    return normalized


def folder_file_pattern(output_folder: str) -> str | None:
    if not output_folder:
        return None
    return rf"^{re.escape(output_folder)}/.*"


def authenticate_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The 'kaggle' package is not installed. Install it with: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def download_kernel_output_folder(
    kernel: str,
    output_folder: str,
    download_dir: Path,
    force: bool,
    quiet: bool,
    page_token: str | None,
) -> list[Path]:
    api = authenticate_kaggle()
    file_pattern = folder_file_pattern(output_folder)
    downloaded_files: list[Path] = []
    seen_tokens: set[str] = set()
    next_page_token = page_token

    while True:
        files, returned_page_token = api.kernels_output(
            kernel,
            path=str(download_dir),
            file_pattern=file_pattern,
            force=force,
            quiet=quiet,
            page_token=next_page_token,
        )
        downloaded_files.extend(Path(download_dir, file_name) for file_name in files)

        if not returned_page_token or returned_page_token in seen_tokens:
            break

        seen_tokens.add(returned_page_token)
        next_page_token = returned_page_token
        if not quiet:
            print(f"Continuing with Kaggle output page token: {returned_page_token}")

    return downloaded_files


def locate_downloaded_folder(download_dir: Path, output_folder: str, downloaded_files: list[Path]) -> Path:
    if not output_folder:
        return download_dir

    expected = download_dir / Path(output_folder)
    if expected.exists():
        return expected

    candidates = [
        path
        for path in downloaded_files
        if output_folder.replace("/", os.sep) in str(path)
    ]
    if candidates:
        parts = Path(output_folder).parts
        for parent in candidates[0].parents:
            if parent.parts[-len(parts):] == parts:
                return parent

    raise FileNotFoundError(
        f"Downloaded output folder '{output_folder}' was not found under '{download_dir}'. "
        "Check --kernel and --output-folder."
    )


def install_folder(source_dir: Path, install_dir: Path, force: bool) -> None:
    if install_dir.exists():
        if not force:
            raise FileExistsError(f"Install directory already exists: {install_dir}. Use --force to replace it.")
        if install_dir.is_dir():
            shutil.rmtree(install_dir)
        else:
            install_dir.unlink()

    install_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, install_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="Kaggle notebook slug, for example 'owner/notebook-slug'.")
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder path inside the Kaggle notebook output, for example 'data/normalized/realwaste'.",
    )
    parser.add_argument("--install-dir", type=Path, required=True, help="Local directory to install the folder into.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Temporary download directory. Defaults to a system temp directory.",
    )
    parser.add_argument("--page-token", default=None, help="Optional Kaggle output page token to start from.")
    parser.add_argument("--force", action="store_true", help="Replace existing downloads/install directory.")
    parser.add_argument("--keep-download", action="store_true", help="Keep the intermediate downloaded output files.")
    parser.add_argument("--quiet", action="store_true", help="Reduce Kaggle API output.")
    
    
    parser.add_argument("--test", type=Path, required=False, help="indicate whether on test or not.")
    return parser.parse_args()


def search_by_pattern(pattern: str, files: list[str]) -> list[str]:
    regex = re.compile(pattern)
    return [file for file in files if regex.match(file)]


def test_run():
    import kaggle
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    configure_utf8_streams()
    api = authenticate_kaggle()
    res = api.kernels_list_files(
        "karimaouaouda/yolo-train-pipeline",
    )
    
    request = ApiListKernelSessionOutputRequest()
    request.user_name = "karimaouaouda"
    request.kernel_slug = "yolo-train-pipeline"
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.list_kernel_session_output(request)
        
    next_token = response.next_page_token
    i = 1
    while next_token:
        print(f"- page {i}:")
        print(f"files : {len(response.files)}, last file : {response.files[-1].fileName}")
        i += 1
        request.page_token = next_token
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            
            next_token = response.next_page_token
            
            
    print(f"files : {len(response.files)}, with token : {response.next_page_token}")    
    return res

def main() -> None:
    configure_utf8_streams()
    args = parse_args()
    if args.test:
        print("Running in test mode. No files will be downloaded or installed.")
        test_run()
        return
    output_folder = normalize_output_folder(args.output_folder)

    if args.download_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="kaggle-output-")
        download_dir = Path(temp_dir.name)
    else:
        temp_dir = None
        download_dir = args.download_dir
        if download_dir.exists() and args.force:
            shutil.rmtree(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

    try:
        downloaded_files = download_kernel_output_folder(
            kernel=args.kernel,
            output_folder=output_folder,
            download_dir=download_dir,
            force=args.force,
            quiet=args.quiet,
            page_token=args.page_token,
        )
        if not downloaded_files:
            raise FileNotFoundError(f"No files were downloaded for output folder '{output_folder}'.")

        source_dir = locate_downloaded_folder(download_dir, output_folder, downloaded_files)
        install_folder(source_dir, args.install_dir, force=args.force)
        print(f"Installed Kaggle output folder '{output_folder}' from '{args.kernel}' to '{args.install_dir}'.")
        print(f"Downloaded files: {len(downloaded_files)}")
    finally:
        if temp_dir is not None and not args.keep_download:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
