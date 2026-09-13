"""Load configurable per-society planning rules from YAML.

The rules are used by the feasibility engine and the layout placer. Unknown
societies fall back to the Generic Pakistan Residential ruleset.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import yaml

SOCIETIES_DIR = Path(__file__).parent / "societies"
DEFAULT_SOCIETY = "generic"

ALIASES = {
    "generic": "generic",
    "generic_pakistan": "generic",
    "pakistan": "generic",
    "dha": "dha",
    "bahria": "bahria",
    "bahria_town": "bahria",
    "cda": "cda",
    "islamabad": "cda",
}


@lru_cache
def _load(name: str) -> Dict[str, Any]:
    path = SOCIETIES_DIR / f"{name}.yaml"
    if not path.exists():
        path = SOCIETIES_DIR / f"{DEFAULT_SOCIETY}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rules(society: str | None) -> Dict[str, Any]:
    key = ALIASES.get((society or "").strip().lower(), DEFAULT_SOCIETY)
    return _load(key)


def list_societies() -> list[str]:
    return sorted({p.stem for p in SOCIETIES_DIR.glob("*.yaml")})
