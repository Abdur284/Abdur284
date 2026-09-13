# PRD — HAVEN AI 2D Floor Plan Design Module

## Goal
Given a plot and a natural-language brief, produce a **feasibility-checked, Pakistan-oriented 2D floor plan** (max 2 stories) rendered with **Stable Diffusion + ControlNet**.

## User story
> "I have a 5 marla plot (25 × 50 ft). I want 3 bedrooms, 2 bathrooms, one kitchen, drawing room, TV lounge, car parking and stairs — double-story."

The system must respond with either:
1. **FEASIBLE** — full per-floor room list, structured layout, control image, generated 2D plan, and a plain-English explanation, **or**
2. **NOT_FEASIBLE** — the exact constraint that fails, and a concrete alternative (typically: reduce rooms, or split across two floors — never more).

## Non-goals
- 3D visualisation.
- Cost estimation / BOQ.
- MEP layouts (plumbing, electrical).
- Automatic verification against official Pakistani society regulations (rules ship as *conceptual* defaults; users must verify against actual bylaws).

## Success criteria
- Feasibility verdict is deterministic and reproducible.
- Layout validator flags overlaps and out-of-bounds placement before any image is generated.
- The generated SD image visually matches the structured layout (rooms match count and rough position).
- 2-story maximum is enforced in code (`max_floors <= 2`).
- Chatbot and web UI hit the same API — feasibility is never bypassed.

## Out of scope for this milestone
- User authentication, save/load projects, sharing.
- Fine-tuned SD checkpoint (we use `runwayml/stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-mlsd`).
- Real regulation ingestion.

## Test surface — see `backend/tests/test_pipeline.py`
1. 5 marla / 10 marla / custom dimensions.
2. Single-story / double-story.
3. Too many rooms → NOT_FEASIBLE with suggestion.
4. Invalid / missing dimensions.
5. Conflicting requirements (huge program on tight setbacks).
6. Parking + staircase requirements.
7. SD failure / ControlNet failure / backend model failure (all captured by the stub path).
8. Ambiguous prompt (creative brief).
9. Overlap detection.
10. Out-of-bounds detection.
