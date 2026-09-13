"""Natural-language brief -> structured `Requirements`.

Two backends:
- `GeminiExtractor` (used automatically when `GEMINI_API_KEY` is set) asks Gemini
  to return strict JSON matching our schema. The LLM is only responsible for
  parsing text; numeric feasibility and layout live in deterministic code.
- `RegexExtractor` (default fallback) parses common Pakistani-brief phrasings:
  "5 marla", "25 x 50 feet", "3 bedrooms with attached bath", "double story", etc.

Both return `Requirements`. If dimensions are missing/ambiguous, we surface it as
a validation error rather than guessing.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from ..schemas import PlotDimensions, Requirements

log = logging.getLogger(__name__)

# 1 marla ≈ 225 sqft (Punjab urban convention). Kept explicit so we don't
# accidentally use the 272-sqft variant.
MARLA_SQFT = 225.0

# Common Pakistani plot presets (width x length in feet) used only when the
# user gives marla without explicit dimensions.
MARLA_PRESETS_FT = {
    3: (20, 34),
    5: (25, 45),
    7: (30, 52),
    8: (30, 60),
    10: (35, 65),
    12: (40, 68),
    14: (40, 75),
    20: (50, 90),
}


class MissingDimensionsError(ValueError):
    pass


# --------------------------- Regex fallback ---------------------------

_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "single": 1, "double": 2, "triple": 3,
}


def _to_int(tok: str) -> Optional[int]:
    tok = tok.lower().strip()
    if tok.isdigit():
        return int(tok)
    return _NUMBERS.get(tok)


def _find_count(prompt: str, keywords: list[str]) -> int:
    """Count instances like '3 bedroom', 'two bathrooms', 'a kitchen'."""
    p = prompt.lower()
    total = 0
    for kw in keywords:
        # "N kw" or "N kws"
        for m in re.finditer(rf"(\b\w+\b)\s+{kw}s?\b", p):
            n = _to_int(m.group(1))
            if n is not None:
                total = max(total, n)
        # bare mention -> at least 1
        if re.search(rf"\b{kw}s?\b", p) and total == 0:
            total = 1
    return total


def _find_bool(prompt: str, keywords: list[str]) -> bool:
    p = prompt.lower()
    return any(re.search(rf"\b{k}\b", p) for k in keywords)


def _extract_plot(prompt: str, override: Optional[PlotDimensions]) -> PlotDimensions:
    if override is not None:
        return override
    p = prompt.lower().replace("×", "x").replace("*", "x").replace("by", "x")

    # "25 x 50 feet"
    m = re.search(r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:ft|feet|foot)?", p)
    if m:
        return PlotDimensions(width_ft=float(m.group(1)), length_ft=float(m.group(2)))

    # "N marla" -> preset (approximation; user should confirm exact dims)
    m = re.search(r"(\d+(?:\.\d+)?)\s*marla", p)
    if m:
        marla = float(m.group(1))
        preset = MARLA_PRESETS_FT.get(int(round(marla)))
        if preset:
            return PlotDimensions(width_ft=float(preset[0]), length_ft=float(preset[1]))
        # Fall back to a square-ish plot of the right area.
        area = marla * MARLA_SQFT
        side = area ** 0.5
        return PlotDimensions(width_ft=round(side, 1), length_ft=round(area / side, 1))

    raise MissingDimensionsError(
        "Plot dimensions not found in the prompt. Please provide width x length in feet "
        "or the marla size (e.g. '5 marla' or '25 x 50 feet')."
    )


def _extract_floors(prompt: str) -> Optional[int]:
    p = prompt.lower()
    if re.search(r"\b(double|two|2)[- ]stor(?:e?y|ies)\b", p):
        return 2
    if re.search(r"\b(single|one|1)[- ]stor(?:e?y|ies)\b", p):
        return 1
    if re.search(r"\bground\s*\+\s*(1|one)\b", p):
        return 2
    return None


def regex_extract(prompt: str, plot: Optional[PlotDimensions] = None, society: str = "generic") -> Requirements:
    plot_dims = _extract_plot(prompt, plot)

    bedrooms = _find_count(prompt, ["bedroom", "bed room", "br"])
    bathrooms = _find_count(prompt, ["bathroom", "washroom", "toilet", "bath"])
    if bathrooms == 0 and re.search(r"attached\s*(?:bath|washroom|bathroom)", prompt.lower()):
        # "3 bedrooms with attached bath" -> one bath per bedroom + 1 common.
        bathrooms = max(bedrooms, 1) + (1 if bedrooms >= 2 else 0)
    kitchens = _find_count(prompt, ["kitchen"])
    guest = _find_count(prompt, ["guest room", "guest"])

    return Requirements(
        plot=plot_dims,
        floors_requested=_extract_floors(prompt),
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        kitchens=max(kitchens, 1) if any([bedrooms, guest]) else kitchens,
        drawing_room=_find_bool(prompt, ["drawing room", "drawing", "sitting room"]),
        tv_lounge=_find_bool(prompt, ["tv lounge", "family lounge", "lounge", "tv room"]),
        guest_room=guest,
        servant_quarter=_find_count(prompt, ["servant quarter", "servant"]),
        parking=_find_bool(prompt, ["parking", "car porch", "garage"]),
        stairs=_find_bool(prompt, ["stair", "staircase"]) or (_extract_floors(prompt) == 2),
        terrace=_find_bool(prompt, ["terrace", "balcony", "rooftop"]),
        society=society,
        raw_prompt=prompt,
    )


# --------------------------- Gemini backend ---------------------------

_GEMINI_SYSTEM = (
    "You extract Pakistani house-planning requirements from the user's brief. "
    "Return STRICT JSON with keys: plot{width_ft,length_ft}, floors_requested (1|2|null), "
    "bedrooms, bathrooms, kitchens, drawing_room (bool), tv_lounge (bool), guest_room, "
    "servant_quarter, parking (bool), stairs (bool), terrace (bool), other (array of strings). "
    "Use feet. 1 marla = 225 sqft (Punjab urban). If the user gave marla without exact "
    "dimensions, propose the standard preset dims for that marla size. Never invent numbers "
    "when the user's brief is silent — use 0 / false / null. Reply with JSON only."
)


def gemini_extract(prompt: str, plot: Optional[PlotDimensions], society: str) -> Requirements:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    import google.generativeai as genai  # local import so the dep is optional
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        system_instruction=_GEMINI_SYSTEM,
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content(prompt)
    data = json.loads(resp.text)

    # Merge structured plot override if provided.
    if plot is not None:
        data["plot"] = plot.model_dump()
    data.setdefault("plot", {})
    data.setdefault("other", [])
    data["society"] = society
    data["raw_prompt"] = prompt
    try:
        return Requirements(**data)
    except Exception as e:  # pragma: no cover — defensive
        log.warning("Gemini returned malformed JSON, falling back to regex: %s", e)
        return regex_extract(prompt, plot, society)


# --------------------------- Public entry ---------------------------

def extract(prompt: str, plot: Optional[PlotDimensions] = None, society: str = "generic") -> Requirements:
    if os.getenv("GEMINI_API_KEY"):
        try:
            return gemini_extract(prompt, plot, society)
        except MissingDimensionsError:
            raise
        except Exception as e:
            log.warning("Gemini extraction failed, using regex fallback: %s", e)
    return regex_extract(prompt, plot, society)
