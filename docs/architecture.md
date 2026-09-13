# Architecture — HAVEN AI 2D Floor Plan Module

## Pipeline (single source of truth)

```
User Prompt (+ optional plot dims + society)
      │
      ▼
Requirement Extraction        (Gemini if key set, deterministic regex fallback)
      │
      ▼
Feasibility Engine            (deterministic; society rules loaded from YAML)
      │
      ▼
[if 1-story infeasible → ask user to accept 2-story; NEVER > 2 floors]
      │
      ▼
Structured Layout             (deterministic pack; validates no-overlap + in-bounds)
      │
      ▼
Control Image                 (PIL: plot boundary, setback, walls, doors, windows, stairs, parking)
      │
      ▼
Stable Diffusion + ControlNet (Colab + ngrok in dev; stub when unavailable)
      │
      ▼
DesignResponse                (feasibility verdict, layout JSON, control + generated PNG, explanation)
```

The **StructuredLayout is the source of truth.** Stable Diffusion may only render the visual — it never adds, moves, or removes rooms. The generated image is compared side-by-side with the structured layout; the frontend surfaces both.

## Modules

| Layer     | Path                                | Responsibility                                                 |
|-----------|-------------------------------------|----------------------------------------------------------------|
| API       | `backend/app/api/design.py`         | HTTP surface; orchestrates the pipeline in order                |
| Extractor | `backend/app/services/extractor.py` | NL → `Requirements` (Gemini or regex)                          |
| Rules     | `backend/app/rules/`                | Per-society YAML: setbacks, coverage, min room dims, parking   |
| Feasibility | `backend/app/services/feasibility.py` | Deterministic verdict + suggested alternative                |
| Layout    | `backend/app/services/layout.py`    | Grid-scan pack + validator; keeps stair aligned across floors  |
| Control image | `backend/app/services/control_image.py` | PIL renderer for preview + ControlNet input               |
| SD client | `backend/app/services/sd_client.py` | Async HTTPX call to Colab/ngrok; graceful stub on failure      |
| Frontend  | `frontend/app/design/page.tsx`      | Prompt UI, feasibility card, per-floor room list, image viewer |

## Centralized Gemini chatbot integration

The chatbot must POST the user's brief to `POST /api/design/generate`. It must not skip feasibility — the endpoint is the only public surface for the 2D module.

```
Gemini chat
  ├─ intent classifier identifies "2D design"
  └─▶ POST /api/design/generate { prompt, plot?, society?, accept_two_story? }
        └─▶ same pipeline as the web UI
              └─▶ DesignResponse (chatbot displays explanation + generated_image_png_b64)
```

If a `NOT_FEASIBLE` response includes the `ONE_STORY_INSUFFICIENT` issue code, the chatbot asks the user "Should I try a two-story design?" and, on yes, re-submits with `accept_two_story: true`. Two stories are the hard cap.

## Failure modes handled

- Missing / invalid dimensions → 422 with human-readable detail.
- Impossible ask (huge program on tiny plot) → `NOT_FEASIBLE` + `suggested_alternative`.
- 1-story infeasible but 2-story would work → `ONE_STORY_INSUFFICIENT`, awaits user confirmation.
- Layout placement fails after feasibility said yes → returned as `LAYOUT_FAILED` (never silently ignored).
- Overlap / out-of-bounds → caught by validator before any image is generated.
- Gemini unreachable → regex extractor.
- SD/ControlNet unreachable, HTTP error, malformed base64, timeout → control image returned as stub with `sd_stub_reason` explaining what happened.

## Configuration

Everything sensitive lives in env:

```
GEMINI_API_KEY=…            (optional; regex fallback if unset)
GEMINI_MODEL=gemini-2.5-flash
SD_CONTROLNET_URL=…         (Colab ngrok URL)
SD_TIMEOUT_SECONDS=180
CORS_ORIGINS=http://localhost:3000
```

Society rules live in `backend/app/rules/societies/*.yaml`. Adding a society = one YAML file. All non-`generic` files are marked `official: false` because we do not invent regulations.
