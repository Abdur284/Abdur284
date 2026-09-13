"""Deterministic feasibility engine.

Given `Requirements` and a society ruleset, compute:
- buildable rectangle after setbacks
- required floor area (rooms + circulation)
- whether it fits on 1 or 2 stories
- structured issues + human-readable suggestion

The engine is the source of truth for feasibility. The LLM never overrides it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from ..schemas import FeasibilityIssue, FeasibilityReport, Requirements


def _room_requirements(req: Requirements, rules: Dict) -> List[Tuple[str, str, Dict]]:
    """Return the ordered list of (kind, room_name, min_dims) that must be placed."""
    mins = rules["min_rooms"]
    out: List[Tuple[str, str, Dict]] = []

    # A staircase is required for a 2-story house, or if the user explicitly asks.
    stair_needed = req.stairs or (req.floors_requested == 2)

    for i in range(req.bedrooms):
        kind = "master" if i == 0 and req.bedrooms >= 2 else "bedroom"
        name = "Master Bedroom" if kind == "master" else f"Bedroom {i + 1}"
        out.append((kind, name, mins[kind]))
    for i in range(req.guest_room):
        out.append(("guest", f"Guest Room {i + 1}" if req.guest_room > 1 else "Guest Room", mins["guest"]))
    for i in range(req.bathrooms):
        out.append(("bathroom", f"Bathroom {i + 1}", mins["bathroom"]))
    for i in range(max(req.kitchens, 1) if req.bedrooms or req.drawing_room or req.tv_lounge else req.kitchens):
        out.append(("kitchen", "Kitchen" if i == 0 else f"Kitchen {i + 1}", mins["kitchen"]))
    if req.drawing_room:
        out.append(("drawing", "Drawing Room", mins["drawing"]))
    if req.tv_lounge:
        out.append(("tv_lounge", "TV Lounge", mins["tv_lounge"]))
    for i in range(req.servant_quarter):
        out.append(("servant", f"Servant Quarter {i + 1}" if req.servant_quarter > 1 else "Servant Quarter", mins["servant"]))
    if stair_needed:
        out.append(("stair", "Staircase", mins["stair"]))
    if req.parking:
        out.append(("parking", "Car Parking", mins["parking"]))

    return out


def _buildable_rect(req: Requirements, rules: Dict) -> Tuple[float, float, float, float]:
    sb = rules["setbacks"]
    x = sb["side_ft"]
    y = sb["front_ft"]
    w = req.plot.width_ft - 2 * sb["side_ft"]
    h = req.plot.length_ft - sb["front_ft"] - sb["rear_ft"]
    return x, y, max(w, 0.0), max(h, 0.0)


def _required_area(rooms: List[Tuple[str, str, Dict]], rules: Dict) -> float:
    base = sum(r[2]["area"] for r in rooms)
    return base * (1 + rules.get("circulation_pct", 12) / 100.0)


def analyse(req: Requirements, rules: Dict) -> FeasibilityReport:
    issues: List[FeasibilityIssue] = []
    notes: List[str] = []

    # ---- Basic input validation ----
    if req.plot.width_ft < 15 or req.plot.length_ft < 25:
        issues.append(FeasibilityIssue(
            code="PLOT_TOO_SMALL",
            message=f"Plot {req.plot.width_ft}×{req.plot.length_ft} ft is below the practical minimum for a house.",
            suggestion="Provide a plot at least ~15×25 ft (roughly 2 marla).",
        ))
        return FeasibilityReport(feasible=False, verdict="NOT_FEASIBLE", floors=1, issues=issues, notes=notes)

    if req.plot.width_ft > req.plot.length_ft * 2 or req.plot.length_ft > req.plot.width_ft * 4:
        notes.append("Unusual aspect ratio; layout will be constrained. Continuing.")

    bx, by, bw, bh = _buildable_rect(req, rules)
    if bw <= 0 or bh <= 0:
        issues.append(FeasibilityIssue(
            code="SETBACKS_CONSUME_PLOT",
            message="Society setbacks leave no buildable area on this plot.",
            suggestion="Choose a larger plot or a society with smaller setbacks.",
        ))
        return FeasibilityReport(feasible=False, verdict="NOT_FEASIBLE", floors=1, issues=issues, notes=notes)

    buildable_area = bw * bh
    coverage_cap = rules["max_coverage_pct"] / 100.0 * req.plot.area_sqft
    ground_floor_cap = min(buildable_area, coverage_cap)

    if req.bedrooms + req.guest_room + int(req.drawing_room) + int(req.tv_lounge) == 0:
        issues.append(FeasibilityIssue(
            code="NO_ROOMS",
            message="No rooms requested. Nothing to design.",
            suggestion="Include at least one bedroom or living space in the brief.",
        ))
        return FeasibilityReport(feasible=False, verdict="NOT_FEASIBLE", floors=1, issues=issues, notes=notes)

    # ---- Parking requirement per society ----
    parking_rule = rules.get("parking", {})
    if req.plot.area_sqft >= parking_rule.get("required_when_area_sqft_gte", 1e9) and not req.parking:
        notes.append("Parking recommended by society rules; adding one car space to the plan.")
        req = req.model_copy(update={"parking": True})

    rooms_needed = _room_requirements(req, rules)
    needed_area = _required_area(rooms_needed, rules)

    # Also enforce the largest single room fits inside the buildable rect.
    biggest = max((r[2]["w"] * r[2]["h"] for r in rooms_needed), default=0)
    if any((r[2]["w"] > bw and r[2]["h"] > bw) or (r[2]["w"] > bh and r[2]["h"] > bh) for r in rooms_needed):
        issues.append(FeasibilityIssue(
            code="ROOM_TOO_LARGE",
            message="A required room does not fit inside the buildable rectangle in any orientation.",
            suggestion="Increase the plot size or drop the largest room (e.g. drawing room).",
        ))

    # ---- Try single-story ----
    if needed_area <= ground_floor_cap and not issues:
        return FeasibilityReport(
            feasible=True, verdict="FEASIBLE", floors=1, issues=[], notes=notes,
        )

    # ---- Try two-story ----
    two_story_cap = 2 * ground_floor_cap
    max_floors = min(int(rules.get("max_floors", 2)), 2)  # HARD CAP: never more than 2
    if max_floors >= 2 and needed_area <= two_story_cap and not issues:
        return FeasibilityReport(
            feasible=True, verdict="FEASIBLE", floors=2,
            issues=[],
            notes=notes + [
                f"Single story requires ~{needed_area:.0f} sqft but only {ground_floor_cap:.0f} sqft is available on one floor. "
                "A two-story layout is recommended."
            ],
        )

    # ---- Not feasible ----
    reason = (
        f"Requested rooms need ~{needed_area:.0f} sqft including circulation. "
        f"The plot yields only {ground_floor_cap:.0f} sqft per floor after setbacks and coverage, "
        f"so even a 2-story build ({two_story_cap:.0f} sqft) cannot accommodate them."
    )
    issues.append(FeasibilityIssue(code="AREA_INSUFFICIENT", message=reason))

    # Build a concrete suggestion.
    suggestions: List[str] = []
    if req.bedrooms > 2:
        suggestions.append(f"reduce bedrooms to {req.bedrooms - 1}")
    if req.drawing_room and req.tv_lounge:
        suggestions.append("merge drawing room and TV lounge into a single living area")
    if req.guest_room:
        suggestions.append("drop the guest room")
    suggested_alt = (
        "Try: " + ", ".join(suggestions) + "." if suggestions
        else "Reduce room count or increase plot size."
    )

    return FeasibilityReport(
        feasible=False,
        verdict="NOT_FEASIBLE",
        floors=1,
        issues=issues,
        notes=notes,
        suggested_alternative=suggested_alt,
    )
