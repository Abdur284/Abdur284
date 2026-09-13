"""Deterministic layout placer + validator.

Given a feasibility-approved requirement set, produce a `StructuredLayout` whose
rooms:
- lie inside the buildable rectangle (setbacks respected),
- do not overlap one another,
- respect the society's minimum room dimensions,
- carry doors on an interior-facing edge and windows on an exterior-facing edge,
- include a staircase whose footprint is preserved on both floors (2-story only).

The placer uses a simple row-based greedy pack: rooms are sorted by area
(largest first), placed left-to-right on the current row, and a new row starts
when the current row is full. This is intentionally boring — the resulting
control image gives ControlNet a clean, unambiguous target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..schemas import Door, PlotDimensions, Requirements, Room, StructuredLayout, Window


@dataclass
class _Cell:
    kind: str
    name: str
    w: float
    h: float


def _cells(req: Requirements, rules: Dict, floor: int, two_story: bool) -> List[_Cell]:
    """Return the ordered list of rooms to place on a given floor.

    Ground floor gets: drawing, parking, kitchen, tv_lounge, stair, 1 bath, 1 bedroom.
    First floor gets:  remaining bedrooms + baths + stair footprint (matched) + tv/family lounge if not placed.
    """
    mins = rules["min_rooms"]

    def cell(kind: str, name: str) -> _Cell:
        m = mins[kind]
        return _Cell(kind=kind, name=name, w=float(m["w"]), h=float(m["h"]))

    ground: List[_Cell] = []
    first: List[_Cell] = []

    if req.parking:
        ground.append(cell("parking", "Car Parking"))
    if req.drawing_room:
        ground.append(cell("drawing", "Drawing Room"))
    if req.kitchens or req.bedrooms or req.drawing_room or req.tv_lounge:
        ground.append(cell("kitchen", "Kitchen"))
    if req.tv_lounge and not two_story:
        ground.append(cell("tv_lounge", "TV Lounge"))
    stair = cell("stair", "Staircase") if (req.stairs or two_story) else None
    if stair:
        ground.append(stair)

    bed_labels: List[str] = []
    for i in range(req.bedrooms):
        bed_labels.append("Master Bedroom" if i == 0 and req.bedrooms >= 2 else f"Bedroom {i + 1}")
    for _ in range(req.guest_room):
        bed_labels.append("Guest Room")

    baths_left = req.bathrooms

    if two_story:
        # Ground gets 1 bedroom + 1 bath if any; the rest go upstairs.
        if bed_labels:
            b = bed_labels.pop(0)
            ground.append(_Cell("bedroom" if b != "Master Bedroom" else "master", b,
                                float(mins["bedroom"]["w"]) if b != "Master Bedroom" else float(mins["master"]["w"]),
                                float(mins["bedroom"]["h"]) if b != "Master Bedroom" else float(mins["master"]["h"])))
            if baths_left:
                ground.append(cell("bathroom", "Bathroom 1"))
                baths_left -= 1

        # First floor: stair footprint (matched), remaining bedrooms/baths, tv_lounge if requested.
        if stair:
            first.append(_Cell(kind="stair", name="Staircase", w=stair.w, h=stair.h))
        for i, b in enumerate(bed_labels):
            kind = "master" if b == "Master Bedroom" else "bedroom"
            first.append(cell(kind, b))
        for i in range(baths_left):
            first.append(cell("bathroom", f"Bathroom {req.bathrooms - baths_left + i + 1}"))
        if req.tv_lounge:
            first.append(cell("tv_lounge", "Family Lounge"))
        if req.terrace:
            first.append(_Cell(kind="terrace", name="Terrace",
                               w=float(mins["bedroom"]["w"]), h=float(mins["bedroom"]["h"])))
    else:
        # Single-story: everything on ground.
        for b in bed_labels:
            kind = "master" if b == "Master Bedroom" else "bedroom"
            ground.append(cell(kind, b))
        for i in range(baths_left):
            ground.append(cell("bathroom", f"Bathroom {i + 1}"))

    return ground if floor == 0 else first


def _rects_overlap(ax: float, ay: float, aw: float, ah: float,
                   bx: float, by: float, bw: float, bh: float) -> bool:
    return ax < bx + bw - 1e-6 and ax + aw > bx + 1e-6 and ay < by + bh - 1e-6 and ay + ah > by + 1e-6


def _pack(cells: List[_Cell], bx: float, by: float, bw: float, bh: float,
          reserved: List[Room] | None = None) -> Tuple[List[Room], List[_Cell]]:
    """Grid-scan pack. Places each cell at the top-leftmost 1-ft slot that
    doesn't collide with anything already placed (including `reserved`).

    Deterministic, order-stable, and honours reserved rectangles — used to keep
    the staircase footprint identical between ground and first floor.
    """
    placed: List[Room] = list(reserved or [])
    output: List[Room] = []
    unplaced: List[_Cell] = []

    ordered = sorted(cells, key=lambda c: c.w * c.h, reverse=True)

    for c in ordered:
        candidates = [(c.w, c.h)]
        if c.w != c.h:
            candidates.append((c.h, c.w))
        best = None
        for w, h in candidates:
            if w > bw + 1e-6 or h > bh + 1e-6:
                continue
            # Scan the buildable rectangle in 1-ft steps, top row first.
            y = by
            while y + h <= by + bh + 1e-6 and best is None:
                x = bx
                while x + w <= bx + bw + 1e-6:
                    ok = True
                    for p in placed:
                        if _rects_overlap(x, y, w, h, p.x, p.y, p.width, p.height):
                            ok = False
                            # Jump past the obstacle for speed.
                            x = p.x + p.width
                            break
                    if ok:
                        best = (x, y, w, h)
                        break
                    x += 1.0
                y += 1.0
            if best is not None:
                break
        if best is None:
            unplaced.append(c)
            continue
        x, y, w, h = best
        room = Room(room_name=c.name, kind=c.kind, floor=0, x=x, y=y, width=w, height=h)
        placed.append(room)
        output.append(room)

    return output, unplaced


def _attach_openings(rooms: List[Room], bx: float, by: float, bw: float, bh: float) -> None:
    """Add one door and (for habitable rooms) one window per room.

    Windows go on whichever side touches the plot boundary; doors go on the
    opposite side (interior-facing). Bathrooms get no window on the boundary
    (privacy); staircase and parking skip windows.
    """
    for r in rooms:
        touches_n = abs(r.y - by) < 1e-6
        touches_s = abs((r.y + r.height) - (by + bh)) < 1e-6
        touches_w = abs(r.x - bx) < 1e-6
        touches_e = abs((r.x + r.width) - (bx + bw)) < 1e-6

        # Door: pick an interior-facing side; if all sides are on the boundary
        # (small plots), place the door on the south side by default.
        door_side = "S"
        for side, touches in (("N", touches_n), ("S", touches_s), ("W", touches_w), ("E", touches_e)):
            if not touches:
                door_side = side
                break
        door_span = r.width if door_side in ("N", "S") else r.height
        r.doors.append(Door(side=door_side, offset_ft=max(door_span / 2 - 1.5, 0.0), width_ft=min(3.0, door_span)))

        # Window: skip for stairs, parking, bathrooms.
        if r.kind in ("stair", "parking", "bathroom", "corridor"):
            continue
        window_side = None
        for side, touches in (("N", touches_n), ("S", touches_s), ("W", touches_w), ("E", touches_e)):
            if touches:
                window_side = side
                break
        if window_side is None:
            continue
        window_span = r.width if window_side in ("N", "S") else r.height
        r.windows.append(Window(side=window_side, offset_ft=max(window_span / 2 - 2.0, 0.0), width_ft=min(4.0, window_span)))


def validate(layout: StructuredLayout) -> List[str]:
    """Return a list of validation errors. Empty list = layout is valid."""
    errs: List[str] = []
    bx, by = layout.buildable_x, layout.buildable_y
    bw, bh = layout.buildable_w, layout.buildable_h

    for f in range(layout.floors):
        rooms = layout.rooms_on(f)
        for r in rooms:
            if r.x < bx - 1e-6 or r.y < by - 1e-6 or r.x + r.width > bx + bw + 1e-6 or r.y + r.height > by + bh + 1e-6:
                errs.append(f"Floor {f}: '{r.room_name}' is outside the buildable area.")
        # Pairwise overlap check.
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                if (a.x < b.x + b.width - 1e-6 and a.x + a.width > b.x + 1e-6
                        and a.y < b.y + b.height - 1e-6 and a.y + a.height > b.y + 1e-6):
                    errs.append(f"Floor {f}: rooms '{a.room_name}' and '{b.room_name}' overlap.")

    # Staircase must exist and align between floors on 2-story designs.
    if layout.floors == 2:
        g_stair = next((r for r in layout.rooms_on(0) if r.kind == "stair"), None)
        f_stair = next((r for r in layout.rooms_on(1) if r.kind == "stair"), None)
        if not g_stair or not f_stair:
            errs.append("Two-story plan is missing a staircase on one of the floors.")
        elif (abs(g_stair.x - f_stair.x) > 0.5 or abs(g_stair.y - f_stair.y) > 0.5
              or abs(g_stair.width - f_stair.width) > 0.5 or abs(g_stair.height - f_stair.height) > 0.5):
            errs.append("Staircase footprint does not align between ground and first floor.")

    return errs


def build_layout(req: Requirements, rules: Dict, floors: int) -> StructuredLayout:
    sb = rules["setbacks"]
    bx, by = sb["side_ft"], sb["front_ft"]
    bw = req.plot.width_ft - 2 * sb["side_ft"]
    bh = req.plot.length_ft - sb["front_ft"] - sb["rear_ft"]

    # Reserve the staircase footprint at a deterministic corner (top-right of
    # the buildable rectangle) so it's identical on both floors.
    stair_reserved: List[Room] = []
    if floors == 2 or req.stairs:
        m = rules["min_rooms"]["stair"]
        sw, sh = float(m["w"]), float(m["h"])
        stair_x = bx + bw - sw
        stair_y = by
        if stair_x < bx or stair_y + sh > by + bh + 1e-6:
            raise ValueError("Buildable area too small to host a staircase.")
        stair_reserved.append(Room(
            room_name="Staircase", kind="stair", floor=0,
            x=stair_x, y=stair_y, width=sw, height=sh,
        ))

    all_rooms: List[Room] = []
    for f in range(floors):
        cells = _cells(req, rules, floor=f, two_story=(floors == 2))
        # Drop the stair cell from the pack list — we place it as a reserved room.
        cells = [c for c in cells if c.kind != "stair"]

        reserved_this_floor: List[Room] = []
        if stair_reserved:
            r0 = stair_reserved[0]
            reserved_this_floor.append(Room(
                room_name=r0.room_name, kind=r0.kind, floor=f,
                x=r0.x, y=r0.y, width=r0.width, height=r0.height,
            ))

        placed, unplaced = _pack(cells, bx, by, bw, bh, reserved=reserved_this_floor)
        if unplaced:
            names = ", ".join(f"{c.name} ({c.w}x{c.h}ft)" for c in unplaced)
            raise ValueError(f"Layout placement failed on floor {f}: could not place {names}.")
        # Combine the reserved stair + packed rooms and stamp the floor number.
        floor_rooms = reserved_this_floor + placed
        for r in floor_rooms:
            r.floor = f
        _attach_openings(floor_rooms, bx, by, bw, bh)
        all_rooms.extend(floor_rooms)

    layout = StructuredLayout(
        plot=req.plot,
        setback_front_ft=sb["front_ft"], setback_rear_ft=sb["rear_ft"], setback_side_ft=sb["side_ft"],
        buildable_x=bx, buildable_y=by, buildable_w=bw, buildable_h=bh,
        floors=floors, rooms=all_rooms,
    )

    errs = validate(layout)
    if errs:
        raise ValueError("Layout validation failed: " + "; ".join(errs))
    return layout
