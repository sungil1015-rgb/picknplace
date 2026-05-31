"""zip → raw/<날짜>_<출처>/ 자동 풀기 + 평탄화 + _v2 suffix.

Used by GUI's "+ zip import" button and CLI scripts.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
import shutil
from .data_manager import DatasetMeta, write_meta


REQUIRED_SUFFIXES = (".bmp", ".ply", "_data.json", "_info.json")


def _is_required(name: str) -> bool:
    return any(name.endswith(suf) for suf in REQUIRED_SUFFIXES)


def import_dataset_zip(
    zip_path: Path,
    raw_root: Path,
    source: str,
    date: str,
) -> Path:
    """Extract zip into raw_root/<date>_<source>/ flat. Returns target dir.

    - Flattens nested directories: any 4-tuple file inside any subdir of zip
      is placed at top of target.
    - If target exists, appends _v2, _v3, ... suffix.
    - Validates: at least 1 complete 4-tuple (BMP+PLY+_data.json+_info.json).
    - Seeds meta.yaml.

    Args:
        zip_path: Path to zip file.
        raw_root: Drive raw/ folder.
        source: Source name (team1, team2, self, ...).
        date: YYYYMMDD string.

    Raises:
        ValueError if no complete 4-tuple in zip (target rolled back).
    """
    zip_path = Path(zip_path)
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    base_name = f"{date}_{source}"
    target = raw_root / base_name
    suffix = 2
    while target.exists():
        target = raw_root / f"{base_name}_v{suffix}"
        suffix += 1
    target.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename
                if info.is_dir():
                    continue
                base = Path(name).name  # strip nested dirs
                if not base or not _is_required(base):
                    continue
                with zf.open(info) as src, open(target / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # validate at least one complete 4-tuple
        bmps = list(target.glob("*.bmp"))
        valid_shots = []
        for bmp in bmps:
            stem = bmp.stem
            if all((target / f"{stem}{suf}").exists()
                   if suf.startswith("_")
                   else (target / f"{stem}{suf}").exists()
                   for suf in [".ply", "_data.json", "_info.json"]):
                valid_shots.append(stem)

        if not valid_shots:
            raise ValueError(
                f"zip has no complete shots "
                f"(BMP+PLY+_data.json+_info.json 4쌍 없음): {zip_path}"
            )

        # seed meta.yaml
        shot_date_iso = (
            f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            if len(date) == 8 and date.isdigit()
            else date
        )
        meta = DatasetMeta(
            source=source, shot_date=shot_date_iso,
            camera="Zivid 2+ MR60", serial="",
            n_shots=len(valid_shots), n_polygons=0,
            labeled_by="", prelabel_method="manual",
            prelabel_box_roi=None,
            notes=f"Imported from {zip_path.name}",
        )
        write_meta(target, meta)
        return target

    except Exception:
        # rollback partial extraction
        if target.exists():
            shutil.rmtree(target)
        raise
