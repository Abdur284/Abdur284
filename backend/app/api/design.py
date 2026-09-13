"""HTTP surface for the 2D Floor Plan Design module.

Pipeline order matches the spec:
    prompt
      -> requirement extraction (Gemini or regex fallback)
      -> feasibility engine (deterministic)
      -> [optional retry as 2-story if the user confirmed]
      -> structured layout (deterministic pack + validate)
      -> control image (deterministic PIL render)
      -> Stable Diffusion + ControlNet (Colab via ngrok, or stub)
      -> DesignResponse

The centralized Gemini chatbot is expected to call POST /api/design/generate
directly with the user's brief; it MUST NOT bypass feasibility.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from ..rules.loader import list_societies, load_rules
from ..schemas import DesignRequest, DesignResponse, FeasibilityIssue, FeasibilityReport
from ..services import control_image, extractor, feasibility, layout as layout_svc, sd_client

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/design", tags=["design"])


@router.get("/societies")
def societies():
    return {"societies": list_societies()}


@router.post("/generate", response_model=DesignResponse)
async def generate(req: DesignRequest) -> DesignResponse:
    # --- 1. Requirement extraction ---
    try:
        requirements = extractor.extract(req.prompt, req.plot, req.society)
    except extractor.MissingDimensionsError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.exception("extraction failed")
        raise HTTPException(status_code=400, detail=f"Could not parse the brief: {e}")

    rules = load_rules(requirements.society)

    # --- 2. Feasibility ---
    report = feasibility.analyse(requirements, rules)

    # If a 2-story layout would rescue an infeasible 1-story ask but the user
    # hasn't confirmed yet, return the suggestion without generating an image.
    if report.floors == 2 and requirements.floors_requested != 2 and not req.accept_two_story:
        return DesignResponse(
            requirements=requirements,
            feasibility=FeasibilityReport(
                feasible=False,
                verdict="NOT_FEASIBLE",
                floors=1,
                issues=[FeasibilityIssue(
                    code="ONE_STORY_INSUFFICIENT",
                    message="Your requirements do not fit on a single story on this plot.",
                    suggestion="Confirm you want a two-story house and I will generate it.",
                )],
                notes=report.notes,
                suggested_alternative="Confirm two-story to proceed.",
            ),
            explanation=(
                "Single-story is not feasible. A two-story layout would satisfy the brief on this plot. "
                "Send the same request again with `accept_two_story: true` to generate it."
            ),
        )

    if not report.feasible:
        return DesignResponse(
            requirements=requirements,
            feasibility=report,
            explanation=(
                f"NOT FEASIBLE. { '; '.join(i.message for i in report.issues) } "
                f"{report.suggested_alternative or ''}"
            ).strip(),
        )

    # --- 3. Structured layout ---
    try:
        layout = layout_svc.build_layout(requirements, rules, floors=report.floors)
    except ValueError as e:
        # Feasibility said yes but placement failed — report honestly rather than hiding it.
        return DesignResponse(
            requirements=requirements,
            feasibility=FeasibilityReport(
                feasible=False, verdict="NOT_FEASIBLE", floors=report.floors,
                issues=[FeasibilityIssue(code="LAYOUT_FAILED", message=str(e))],
                notes=report.notes,
                suggested_alternative="Reduce room count or increase plot dimensions.",
            ),
            explanation=f"Layout placement failed: {e}",
        )

    # --- 4. Control image (ground floor is the primary preview) ---
    control_b64 = control_image.render_floor_b64(layout, floor=0, mode="controlnet")
    preview_b64 = control_image.render_floor_b64(layout, floor=0, mode="preview")

    # --- 5. Stable Diffusion + ControlNet ---
    prompt, neg = sd_client.build_prompt(layout, floor=0)
    sd = await sd_client.generate(control_image_b64=control_b64, prompt=prompt, negative_prompt=neg)

    # --- 6. Compose explanation ---
    per_floor = []
    for f in range(layout.floors):
        names = ", ".join(r.room_name for r in layout.rooms_on(f))
        per_floor.append(f"{'Ground Floor' if f == 0 else 'First Floor'}: {names}")
    explanation = (
        f"FEASIBLE on {'a single story' if layout.floors == 1 else 'two stories'} for a "
        f"{requirements.plot.width_ft:.0f}×{requirements.plot.length_ft:.0f} ft plot "
        f"({requirements.plot.area_sqft:.0f} sqft). " + " | ".join(per_floor) + "."
    )
    if sd.stub_reason:
        explanation += f" (Note: {sd.stub_reason})"

    return DesignResponse(
        requirements=requirements,
        feasibility=report,
        layout=layout,
        control_image_png_b64=preview_b64,       # preview shown to the user
        generated_image_png_b64=sd.image_b64,    # SD result, or the control image as stub
        sd_used=sd.used_sd,
        sd_stub_reason=sd.stub_reason,
        explanation=explanation,
    )
