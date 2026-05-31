"""Box ROI persistence in meta.yaml.

Box ROI is axis-aligned rectangle (u_min, u_max, v_min, v_max) in image pixels.
Drawn once per dataset by user, then auto-applied to all shots in that dataset.
"""
from pathlib import Path
from typing import Optional
from .data_manager import read_meta, write_meta, DatasetMeta


def _seed_meta_from_folder_name(folder: Path) -> DatasetMeta:
    """Build a minimal DatasetMeta from folder naming convention <date>_<source>."""
    name = folder.name
    if "_" in name:
        date_part, source = name.split("_", 1)
    else:
        date_part, source = "", name
    # Format date YYYYMMDD → YYYY-MM-DD if it's 8 digits
    if len(date_part) == 8 and date_part.isdigit():
        shot_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
    else:
        shot_date = date_part
    return DatasetMeta(
        source=source, shot_date=shot_date,
        camera="", serial="",
        n_shots=0, n_polygons=0,
        labeled_by="", prelabel_method="manual",
        prelabel_box_roi=None, notes="",
    )


def save_box_roi(folder: Path, u_min: int, u_max: int,
                 v_min: int, v_max: int) -> None:
    """Save axis-aligned box ROI into meta.yaml.

    Creates meta.yaml if missing (with seed values from folder name).
    Preserves all other meta fields.
    """
    folder = Path(folder)
    meta_path = folder / "meta.yaml"
    if meta_path.exists():
        meta = read_meta(folder)
    else:
        meta = _seed_meta_from_folder_name(folder)
    meta.prelabel_box_roi = {
        "u_min": int(u_min), "u_max": int(u_max),
        "v_min": int(v_min), "v_max": int(v_max),
    }
    write_meta(folder, meta)


def load_box_roi(folder: Path) -> Optional[dict]:
    """Load box ROI from meta.yaml. Returns None if meta or roi missing."""
    folder = Path(folder)
    if not (folder / "meta.yaml").exists():
        return None
    return read_meta(folder).prelabel_box_roi
