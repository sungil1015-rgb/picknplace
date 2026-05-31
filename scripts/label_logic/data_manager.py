"""Dataset folder scanning, progress tracking, meta.yaml IO.

Pure logic — no Tkinter / GUI dependency.
Used by both label_tool GUI and CLI scripts (probe_*, build_dataset, etc).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


DEFAULT_CLASSES = ["bottle", "haribo", "mango", "metal_case", "object", "pencil_case"]


@dataclass(frozen=True)
class Shot:
    """One shot = 4 files (bmp + ply + _data.json + _info.json)."""
    stem: str
    bmp: Path
    ply: Path
    data_json: Path
    info_json: Path


@dataclass
class DatasetMeta:
    """Dataset-level metadata stored at <folder>/meta.yaml."""
    source: str                       # team1 / team2 / self / synthetic / ...
    shot_date: str                    # ISO date "2026-04-24"
    camera: str                       # "Zivid 2+ MR60"
    serial: str                       # camera serial
    n_shots: int
    n_polygons: int
    labeled_by: str
    prelabel_method: str              # manual / cmes_polygon_imported / watershed / model_<run>
    prelabel_box_roi: Optional[dict]  # {u_min, u_max, v_min, v_max} or None
    notes: str
    classes: List[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))


def scan_dataset_shots(folder: Path) -> List[Shot]:
    """Return shots in folder that have all 4 required files.

    Looks for *.bmp, then checks for matching .ply / _data.json / _info.json.
    Skips shots with missing files.
    """
    folder = Path(folder)
    out: List[Shot] = []
    if not folder.exists():
        return out
    for bmp in sorted(folder.glob("*.bmp")):
        stem = bmp.stem
        ply = folder / f"{stem}.ply"
        dj = folder / f"{stem}_data.json"
        ij = folder / f"{stem}_info.json"
        if ply.exists() and dj.exists() and ij.exists():
            out.append(Shot(stem=stem, bmp=bmp, ply=ply,
                            data_json=dj, info_json=ij))
    return out


def read_meta(folder: Path) -> DatasetMeta:
    """Read meta.yaml from folder. Raises if file not found."""
    path = Path(folder) / "meta.yaml"
    if not path.exists():
        raise FileNotFoundError(f"meta.yaml not found in {folder}")
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    classes = d.get("classes")
    if classes is None:
        classes = list(DEFAULT_CLASSES)
    return DatasetMeta(
        source=d.get("source", ""),
        shot_date=d.get("shot_date", ""),
        camera=d.get("camera", ""),
        serial=d.get("serial", ""),
        n_shots=d.get("n_shots", 0),
        n_polygons=d.get("n_polygons", 0),
        labeled_by=d.get("labeled_by", ""),
        prelabel_method=d.get("prelabel_method", "manual"),
        prelabel_box_roi=d.get("prelabel_box_roi"),
        notes=d.get("notes", ""),
        classes=classes,
    )


def write_meta(folder: Path, meta: DatasetMeta) -> None:
    """Write meta.yaml to folder. Creates folder if missing."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "meta.yaml"
    d = {
        "source": meta.source,
        "shot_date": meta.shot_date,
        "camera": meta.camera,
        "serial": meta.serial,
        "n_shots": meta.n_shots,
        "n_polygons": meta.n_polygons,
        "classes": list(meta.classes),
        "labeled_by": meta.labeled_by,
        "prelabel_method": meta.prelabel_method,
        "prelabel_box_roi": meta.prelabel_box_roi,
        "notes": meta.notes,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)
