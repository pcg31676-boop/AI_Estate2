"""Local persistence for property and land inspection sessions."""

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


DEFAULT_INSPECTIONS_DIR = Path(__file__).resolve().parent / "inspections"


def save_inspection(metadata, photos, storage_dir=DEFAULT_INSPECTIONS_DIR):
    """Save one inspection record and its optional categorized photo files."""
    inspection_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    inspection_dir = Path(storage_dir) / inspection_id
    photo_dir = inspection_dir / "photos"
    photo_dir.mkdir(parents=True, exist_ok=False)

    saved_photos = []
    for index, photo in enumerate(photos, start=1):
        suffix = Path(photo.get("filename", "")).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        filename = f"photo-{index:02d}{suffix}"
        (photo_dir / filename).write_bytes(photo["content"])
        saved_photos.append({
            "file": f"photos/{filename}",
            "category": str(photo.get("category", "Other")),
            "original_name": Path(photo.get("filename", filename)).name,
        })

    record = {
        **metadata,
        "inspection_id": inspection_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "photos": saved_photos,
    }
    (inspection_dir / "inspection.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def list_inspections(storage_dir=DEFAULT_INSPECTIONS_DIR):
    """Return readable inspection records, newest first."""
    root = Path(storage_dir)
    if not root.is_dir():
        return []

    records = []
    for record_path in root.glob("*/inspection.json"):
        try:
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)