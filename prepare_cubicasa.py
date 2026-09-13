"""Convert an unzipped CubiCasa5k dump into the flat image+.txt caption format
that `finetune_floor_plan.py` expects.

Usage
-----
    # In Colab, after unzipping the CubiCasa5k zip to /content/cubicasa5k:
    python prepare_cubicasa.py \
        --src /content/cubicasa5k \
        --dst /content/dataset \
        --limit 2000

Captions are auto-derived from the SVG labels (room types + counts). No manual
labeling needed. Skip an apartment if either the PNG or SVG is missing.
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


# CubiCasa5k SVG uses class names like "Room Bedroom", "Room Kitchen",
# "Door", "Window", "Wall", etc.
ROOM_CLASSES = {
    "bedroom": "bedroom",
    "master": "master bedroom",
    "bath": "bathroom",
    "toilet": "bathroom",
    "kitchen": "kitchen",
    "living": "drawing room",
    "diningroom": "TV lounge",
    "dining": "TV lounge",
    "hall": "drawing room",
    "closet": "closet",
    "corridor": "corridor",
    "garage": "car parking",
    "storage": "storage",
    "sauna": "sauna",
    "balcony": "balcony",
    "outdoor": "terrace",
}


def _rooms_from_svg(svg_path: Path) -> Counter:
    """Return a Counter of normalised room names found in the CubiCasa SVG."""
    counts: Counter[str] = Counter()
    try:
        tree = ET.parse(svg_path)
    except ET.ParseError:
        return counts
    root = tree.getroot()
    # Any element with class="Room <SomeType>"
    for el in root.iter():
        cls = el.attrib.get("class", "")
        if "Room" not in cls:
            continue
        for token in cls.split():
            key = token.lower()
            if key in ROOM_CLASSES:
                counts[ROOM_CLASSES[key]] += 1
                break
    return counts


def _caption_from_rooms(rooms: Counter, has_door: bool, has_window: bool) -> str:
    """Build a training caption from the room counts."""
    if not rooms:
        return (
            "2D top-down architectural floor plan, blueprint style, "
            "black-and-white line drawing, labeled rooms"
        )
    parts: list[str] = []
    for name, n in sorted(rooms.items(), key=lambda kv: (-kv[1], kv[0])):
        if n == 1:
            parts.append(f"1 {name}")
        else:
            parts.append(f"{n} {name}s")
    inventory = ", ".join(parts)
    extras = []
    if has_door:
        extras.append("doors")
    if has_window:
        extras.append("windows")
    extras_str = (", with " + " and ".join(extras)) if extras else ""
    return (
        f"2D top-down architectural floor plan, residential house, "
        f"{inventory}{extras_str}, blueprint style, black-and-white line drawing, "
        f"labeled rooms, thin wall lines"
    )


def _has_element(svg_path: Path, needle: str) -> bool:
    try:
        content = svg_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return re.search(rf'class="[^"]*\b{needle}\b', content) is not None


def prepare(src: Path, dst: Path, limit: int | None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    # CubiCasa's per-apartment folders live under colorful/, high_quality/, and
    # high_quality_architectural/. We accept any of them.
    apt_dirs = sorted([p for p in src.rglob("model.svg") if p.is_file()])
    print(f"[prep] found {len(apt_dirs)} apartments under {src}")

    for i, svg in enumerate(apt_dirs):
        if limit and written >= limit:
            break
        apt_dir = svg.parent
        # Prefer the scaled png (smaller, uniform); fall back to original.
        png = apt_dir / "F1_scaled.png"
        if not png.exists():
            png = apt_dir / "F1_original.png"
        if not png.exists():
            skipped += 1
            continue

        rooms = _rooms_from_svg(svg)
        caption = _caption_from_rooms(
            rooms,
            has_door=_has_element(svg, "Door"),
            has_window=_has_element(svg, "Window"),
        )

        stem = f"plan_{written:05d}"
        shutil.copyfile(png, dst / f"{stem}.png")
        (dst / f"{stem}.txt").write_text(caption, encoding="utf-8")
        written += 1
        if written % 200 == 0:
            print(f"[prep] {written} written")

    print(f"[prep] done — wrote {written} pairs to {dst} (skipped {skipped})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to the unzipped CubiCasa5k root.")
    ap.add_argument("--dst", required=True, help="Output folder for image+.txt pairs.")
    ap.add_argument("--limit", type=int, default=None, help="Cap the number of pairs (optional).")
    args = ap.parse_args()
    prepare(Path(args.src), Path(args.dst), args.limit)
