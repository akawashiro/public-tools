import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import logging
import sys
from typing import List

from PIL import Image
from PIL.ExifTags import TAGS

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif"}

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".3gp"}

ALL_EXT = IMAGE_EXT | VIDEO_EXT

_logger = logging.getLogger(__name__)


def setup_logging(level: str):
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s"
    )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def get_exif_datetime(path: Path):
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            exif_data = {TAGS.get(tag, tag): value for tag, value in exif.items()}
            for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                if key in exif_data:
                    return datetime.strptime(exif_data[key], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return None


def get_video_datetime(path: Path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "format_tags=creation_time",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        creation = data.get("format", {}).get("tags", {}).get("creation_time")
        if creation:
            return datetime.fromisoformat(creation.replace("Z", "+00:00"))
    except Exception:
        pass
    return None


def get_file_datetime(path: Path):
    return datetime.fromtimestamp(path.stat().st_mtime)


def get_capture_time(path: Path):
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        dt = get_exif_datetime(path)
        if dt:
            return dt
    if ext in VIDEO_EXT:
        dt = get_video_datetime(path)
        if dt:
            return dt
    return get_file_datetime(path)


def format_timestamp(dt: datetime, tz):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    dt = dt.astimezone(tz)
    return dt.strftime("%Y-%m-%dT%H-%M-%S%z")


def parse_date(value: str) -> date:
    try:
        if len(value) != 10 or value[4] != "-" or value[7] != "-":
            raise ValueError
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid date: {value!r} (expected YYYY-MM-DD)"
        ) from error


def sha256_file(path: Path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def collect_files(src: Path, exclude_pattern: str | None = None) -> list[Path]:
    _logger.info(f"Start collecting files under {src}")
    fs: list[Path] = []
    stack = [os.fspath(src)]
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as entries:
            _logger.info(f"Scanning {directory}")
            for entry in entries:
                relative_path = os.path.relpath(entry.path, src)
                if exclude_pattern and (
                    fnmatch.fnmatch(entry.name, exclude_pattern)
                    or fnmatch.fnmatch(relative_path, exclude_pattern)
                ):
                    _logger.info(f"Excluding {entry.path}")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext in ALL_EXT and entry.is_file(follow_symlinks=False):
                    fs.append(Path(entry.path))
    _logger.info(f"End collecting files under {src}")
    return fs


def prepare_file(
    path: Path,
    dst_dir: Path,
    tz,
    since: date | None = None,
) -> tuple[Path, Path, str] | None:
    _logger.info(f"Processing {path}")
    dt = get_capture_time(path)
    local_dt = dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)
    if since is not None and local_dt.date() < since:
        _logger.info(f"SKIP before {since.isoformat()}: {path}")
        return None

    h = sha256_file(path)
    ts = format_timestamp(dt, tz)
    new_name = ts + "-" + h[:8] + path.suffix.lower()
    dst = dst_dir / new_name
    _logger.info(f"{path} -> {dst}")
    return path, dst, h


def copy_file(operation: tuple[Path, Path, str], delete_original: bool):
    path, dst, _ = operation
    shutil.copy(path, dst)
    if delete_original:
        path.unlink()


def process(
    src_dir: Path,
    dst_dir: Path,
    tz,
    dry_run=False,
    delete_original=False,
    exclude_pattern: str | None = None,
    workers: int = 64,
    since: date | None = None,
):
    seen_hashes = set()
    ps = collect_files(src_dir, exclude_pattern)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        operations = executor.map(
            lambda path: prepare_file(path, dst_dir, tz, since),
            ps,
        )

        unique_operations = []
        for operation in operations:
            if operation is None:
                continue
            path, _, h = operation
            if h in seen_hashes:
                _logger.warning(f"SKIP duplicate {path}")
                continue
            seen_hashes.add(h)
            unique_operations.append(operation)

        if not dry_run:
            list(
                executor.map(
                    lambda operation: copy_file(operation, delete_original),
                    unique_operations,
                )
            )


def main():
    parser = argparse.ArgumentParser(
            description="""
Copy photos/videos and rename by capture time.
Example: ./photo_sort.sh --exclude-pattern "@Recently-Snapshot" /media/akira/QNAPTS230/UploadedPictures /media/akira/QNAPTS230/Pictures""",
            formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("source", type=Path, help="source directory")
    parser.add_argument("destination", type=Path, help="destination directory")
    parser.add_argument(
        "--timezone", default="Asia/Tokyo", help="timezone (default: Asia/Tokyo)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show operations without copying"
    )
    parser.add_argument(
        "--exclude-pattern",
        help="exclude files and directories matching this glob pattern",
    )
    parser.add_argument("--delete-original", action="store_true", help="Delete the original file after copying it")
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="number of parallel workers (default: 64)",
    )
    parser.add_argument(
        "--since",
        type=parse_date,
        metavar="YYYY-MM-DD",
        help="process files captured on or after this date",
    )
    parser.add_argument("--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level (default: INFO)")

    args = parser.parse_args()
    src = args.source
    dst = args.destination

    setup_logging(args.log_level)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not src.is_dir():
        raise SystemExit("source must be directory")
    tz = ZoneInfo(args.timezone)
    dst.mkdir(parents=True, exist_ok=True)
    process(
        src,
        dst,
        tz,
        args.dry_run,
        args.delete_original,
        args.exclude_pattern,
        args.workers,
        args.since,
    )


if __name__ == "__main__":
    main()
