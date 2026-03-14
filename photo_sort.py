import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import logging
import sys

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


def sha256_file(path: Path, chunk=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)

    return h.hexdigest()


def collect_files(src: Path):
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALL_EXT:
            yield p


def process(src_dir: Path, dst_dir: Path, tz, dry_run=False, delete_original=False):
    seen_hashes = set()
    for path in collect_files(src_dir):
        h = sha256_file(path)
        if h in seen_hashes:
            _logger.warning(f"SKIP duplicate {path}")
            continue
        seen_hashes.add(h)
        dt = get_capture_time(path)
        ts = format_timestamp(dt, tz)
        new_name = ts + "-" + h[:8] + path.suffix.lower()
        dst = dst_dir / new_name
        _logger.info(f"{path} -> {dst}")

        if not dry_run:
            shutil.copy(path, dst)
        if delete_original:
            shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(
        description="Copy photos/videos and rename by capture time"
    )
    parser.add_argument("source", type=Path, help="source directory")
    parser.add_argument("destination", type=Path, help="destination directory")
    parser.add_argument(
        "--timezone", default="Asia/Tokyo", help="timezone (default: Asia/Tokyo)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show operations without copying"
    )
    parser.add_argument("--delete-original", action="store_true", help="Delete the original file after copying it")
    parser.add_argument("--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="logging level (default: INFO)")

    args = parser.parse_args()
    src = args.source
    dst = args.destination

    setup_logging(args.log_level)
    if not src.is_dir():
        raise SystemExit("source must be directory")
    tz = ZoneInfo(args.timezone)
    dst.mkdir(parents=True, exist_ok=True)
    process(src, dst, tz, args.dry_run, args.delete_original)


if __name__ == "__main__":
    main()
