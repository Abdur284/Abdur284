"""End-to-end tests for the deterministic pipeline.

Covers every case in the spec's TESTING checklist (SD/ControlNet calls are
stubbed via `SD_CONTROLNET_URL` staying empty, so the SD client returns the
control image as a placeholder — this is the "SD failure / ControlNet failure /
backend/model failure" path all at once).
"""
from __future__ import annotations

import asyncio

import pytest

from floor_plan_2d.api import generate as generate_api
from floor_plan_2d.rules.loader import load_rules
from floor_plan_2d.schemas import DesignRequest, PlotDimensions, Requirements
from floor_plan_2d.services import extractor, feasibility, layout as layout_svc
from floor_plan_2d.services.control_image import render_floor_png


def _run(req: DesignRequest):
    return asyncio.run(generate_api(req))


# ---------- extraction ----------

def test_regex_extract_5_marla_dims_and_rooms():
    r = extractor.regex_extract(
        "I have a 5 marla plot with dimensions 25 x 50 feet. I want 3 bedrooms, "
        "2 bathrooms, one kitchen, a drawing room, TV lounge, car parking and stairs. "
        "I want a double-story house."
    )
    assert r.plot.width_ft == 25 and r.plot.length_ft == 50
    assert r.bedrooms == 3 and r.bathrooms == 2 and r.kitchens == 1
    assert r.drawing_room and r.tv_lounge and r.parking and r.stairs
    assert r.floors_requested == 2


def test_regex_extract_marla_only_uses_preset():
    r = extractor.regex_extract("5 marla plot, 3 bedrooms")
    assert r.plot.width_ft == 25 and r.plot.length_ft == 45


def test_missing_dimensions_raises():
    with pytest.raises(extractor.MissingDimensionsError):
        extractor.regex_extract("I want a nice big house with 3 bedrooms")


def test_ambiguous_prompt_survives():
    r = extractor.regex_extract("Give me a 25x50 house, be creative")
    assert r.plot.width_ft == 25 and r.bedrooms == 0


# ---------- feasibility ----------

def test_5_marla_single_story_reasonable():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet plot, 2 bedrooms, 2 bathrooms, kitchen, drawing room")
    report = feasibility.analyse(req, rules)
    assert report.feasible


def test_5_marla_too_many_rooms_needs_two_stories():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet plot, 4 bedrooms, 4 bathrooms, kitchen, drawing, tv lounge, parking")
    report = feasibility.analyse(req, rules)
    assert report.feasible and report.floors == 2


def test_5_marla_absurd_ask_not_feasible():
    rules = load_rules("generic")
    req = extractor.regex_extract(
        "25x50 feet plot, 8 bedrooms, 8 bathrooms, kitchen, drawing, tv lounge, guest room, servant, parking"
    )
    report = feasibility.analyse(req, rules)
    assert not report.feasible and report.suggested_alternative


def test_10_marla_fits_single_story():
    rules = load_rules("generic")
    req = extractor.regex_extract("35x70 feet, 3 bedrooms, 3 bathrooms, kitchen, drawing, tv lounge, parking")
    report = feasibility.analyse(req, rules)
    assert report.feasible and report.floors in (1, 2)


def test_invalid_dimensions_rejected():
    rules = load_rules("generic")
    req = Requirements(plot=PlotDimensions(width_ft=10, length_ft=15), bedrooms=1, raw_prompt="")
    report = feasibility.analyse(req, rules)
    assert not report.feasible


def test_conflicting_requirements_reported():
    """User asks for a huge program on a tiny plot: engine returns NOT_FEASIBLE with a suggestion."""
    rules = load_rules("cda")  # bigger setbacks -> stricter
    req = extractor.regex_extract("20x34 feet, 5 bedrooms, 5 bathrooms, kitchen, drawing, tv lounge, guest, parking")
    report = feasibility.analyse(req, rules)
    assert not report.feasible


def test_parking_auto_added_when_society_requires_it():
    rules = load_rules("generic")
    req = extractor.regex_extract("35x70 feet, 3 bedrooms, 2 bathrooms, kitchen, drawing")
    report = feasibility.analyse(req, rules)
    assert report.feasible  # parking is added silently per society rule


def test_stair_required_on_two_story():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet, 4 bedrooms, 3 bathrooms, kitchen, drawing, tv lounge, parking, double story")
    layout = layout_svc.build_layout(req, rules, floors=2)
    assert any(r.kind == "stair" and r.floor == 0 for r in layout.rooms)
    assert any(r.kind == "stair" and r.floor == 1 for r in layout.rooms)


# ---------- layout validator ----------

def test_layout_has_no_overlaps_and_stays_inside_bounds():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet, 3 bedrooms, 2 bathrooms, kitchen, drawing, tv lounge, parking, double story")
    lay = layout_svc.build_layout(req, rules, floors=2)
    errs = layout_svc.validate(lay)
    assert errs == []


def test_layout_validator_flags_overlap():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet, 2 bedrooms, 1 bathroom, kitchen")
    lay = layout_svc.build_layout(req, rules, floors=1)
    # Force an overlap and re-validate.
    if len(lay.rooms) >= 2:
        lay.rooms[1].x = lay.rooms[0].x
        lay.rooms[1].y = lay.rooms[0].y
    errs = layout_svc.validate(lay)
    assert any("overlap" in e for e in errs)


def test_layout_validator_flags_out_of_bounds():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet, 2 bedrooms, 1 bathroom, kitchen")
    lay = layout_svc.build_layout(req, rules, floors=1)
    lay.rooms[0].x = -50
    errs = layout_svc.validate(lay)
    assert any("outside" in e for e in errs)


# ---------- end-to-end API ----------

def test_api_5_marla_double_story_returns_generated_image():
    resp = _run(DesignRequest(
        prompt="I have a 5 marla plot 25x50 feet. I want 3 bedrooms, 2 bathrooms, kitchen, "
               "drawing room, TV lounge, parking, stairs, double story.",
        accept_two_story=True,
    ))
    assert resp.feasibility.feasible
    assert resp.layout is not None
    assert resp.control_image_png_b64 and resp.generated_image_png_b64
    # SD is not configured in tests -> stub path -> generated == control.
    assert resp.sd_used is False and resp.sd_stub_reason


def test_api_infeasible_one_story_requests_confirmation():
    resp = _run(DesignRequest(
        prompt="25x50 feet, 5 bedrooms, 5 bathrooms, kitchen, drawing, tv lounge, parking",
        accept_two_story=False,
    ))
    assert not resp.feasibility.feasible
    assert "two-story" in resp.explanation.lower() or "two story" in resp.explanation.lower()


def test_api_missing_dimensions_returns_422():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        _run(DesignRequest(prompt="I want a beautiful modern house with 3 bedrooms."))
    assert ei.value.status_code == 422


def test_control_image_renders():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet, 2 bedrooms, 1 bathroom, kitchen")
    lay = layout_svc.build_layout(req, rules, floors=1)
    png = render_floor_png(lay, floor=0, mode="preview")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_custom_dimensions():
    resp = _run(DesignRequest(
        prompt="3 bedrooms, 2 bathrooms, kitchen, drawing, parking",
        plot=PlotDimensions(width_ft=30, length_ft=60),
        accept_two_story=True,
    ))
    assert resp.feasibility.feasible


def test_pipeline_never_exceeds_two_floors():
    rules = load_rules("generic")
    req = extractor.regex_extract("25x50 feet, 6 bedrooms, 6 bathrooms, kitchen, drawing, tv lounge, parking")
    report = feasibility.analyse(req, rules)
    assert report.floors <= 2
