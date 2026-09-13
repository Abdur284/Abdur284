"""Pydantic schemas for the 2D Floor Plan Design pipeline.

The `StructuredLayout` returned by the constraint engine is the source of truth
for every downstream stage (control image, Stable Diffusion + ControlNet, API
response). The generated SD image must not add, remove, or move rooms.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------- Input ----------------------------

class PlotDimensions(BaseModel):
    width_ft: float = Field(..., gt=0, description="Front width of the plot in feet.")
    length_ft: float = Field(..., gt=0, description="Depth of the plot in feet.")

    @property
    def area_sqft(self) -> float:
        return self.width_ft * self.length_ft


class Requirements(BaseModel):
    """Structured requirements extracted from the user's natural-language brief."""

    plot: PlotDimensions
    floors_requested: Optional[int] = Field(None, ge=1, le=2)
    bedrooms: int = 0
    bathrooms: int = 0
    kitchens: int = 0
    drawing_room: bool = False
    tv_lounge: bool = False
    guest_room: int = 0
    servant_quarter: int = 0
    parking: bool = False
    stairs: bool = False
    terrace: bool = False
    other: List[str] = Field(default_factory=list)
    society: str = "generic"
    raw_prompt: str = ""

    @field_validator("society")
    @classmethod
    def _lower(cls, v: str) -> str:
        return (v or "generic").strip().lower()


class DesignRequest(BaseModel):
    """Public request body for POST /api/design/generate."""

    prompt: str = Field(..., min_length=1)
    plot: Optional[PlotDimensions] = None
    society: str = "generic"
    accept_two_story: bool = Field(
        default=False,
        description=(
            "If the single-story attempt is infeasible, retry as 2-story only when this is true. "
            "The frontend flips it to true after the user confirms the 2-story suggestion."
        ),
    )


# ---------------------------- Structured layout ----------------------------

class Door(BaseModel):
    # Position on the room boundary, expressed as (side, offset_ft, width_ft).
    side: Literal["N", "S", "E", "W"]
    offset_ft: float = Field(..., ge=0)
    width_ft: float = Field(default=3.0, gt=0)


class Window(BaseModel):
    side: Literal["N", "S", "E", "W"]
    offset_ft: float = Field(..., ge=0)
    width_ft: float = Field(default=4.0, gt=0)


class Room(BaseModel):
    room_name: str
    kind: str  # bedroom, bathroom, kitchen, drawing, tv_lounge, stair, parking, ...
    floor: int  # 0 = ground, 1 = first
    x: float
    y: float
    width: float
    height: float
    doors: List[Door] = Field(default_factory=list)
    windows: List[Window] = Field(default_factory=list)

    @property
    def area(self) -> float:
        return self.width * self.height


class StructuredLayout(BaseModel):
    """Deterministic geometry produced by the constraint engine."""

    plot: PlotDimensions
    setback_front_ft: float
    setback_rear_ft: float
    setback_side_ft: float
    buildable_x: float
    buildable_y: float
    buildable_w: float
    buildable_h: float
    floors: int  # 1 or 2
    rooms: List[Room]

    def rooms_on(self, floor: int) -> List[Room]:
        return [r for r in self.rooms if r.floor == floor]


# ---------------------------- Feasibility ----------------------------

class FeasibilityIssue(BaseModel):
    code: str
    message: str
    suggestion: Optional[str] = None


class FeasibilityReport(BaseModel):
    feasible: bool
    verdict: Literal["FEASIBLE", "NOT_FEASIBLE"]
    floors: int
    issues: List[FeasibilityIssue] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    suggested_alternative: Optional[str] = None


# ---------------------------- Response ----------------------------

class DesignResponse(BaseModel):
    requirements: Requirements
    feasibility: FeasibilityReport
    layout: Optional[StructuredLayout] = None
    # base64 PNGs; frontend decodes and displays.
    control_image_png_b64: Optional[str] = None
    generated_image_png_b64: Optional[str] = None
    sd_used: bool = False
    sd_stub_reason: Optional[str] = None
    explanation: str = ""
