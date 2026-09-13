# HAVEN AI — 2D Floor Plan Design Module

Drop-in Python package for HAVEN AI's 2D House / Floor Plan Design module.
No frontend, no standalone backend — just the module code you plug into your
existing HAVEN AI FastAPI app.

## Pipeline (deterministic; SD only renders the visual)

```
user prompt (+ plot dims + society)
  → requirement extraction   (Gemini if GEMINI_API_KEY set, regex fallback)
  → feasibility engine       (deterministic, per-society YAML rules)
  → structured layout        (grid-scan pack + validator; stair aligns across floors)
  → control image            (PIL: walls, doors, windows, stairs, parking)
  → Stable Diffusion + ControlNet   (Colab via ngrok; graceful stub if unavailable)
  → DesignResponse
```

Max 2 stories — hard cap in code.

## Wire into your existing FastAPI backend

```python
from fastapi import FastAPI
from floor_plan_2d import router as floor_plan_router

app = FastAPI()
app.include_router(floor_plan_router)     # exposes /api/design/generate and /api/design/societies
```

Or call the services directly, bypassing HTTP:

```python
from floor_plan_2d.services import extractor, feasibility, layout, control_image, sd_client
from floor_plan_2d.rules.loader import load_rules

req   = extractor.extract("25x50 feet, 3 bed, 2 bath, kitchen, drawing, tv lounge, parking, double story")
rules = load_rules(req.society)
report = feasibility.analyse(req, rules)
if report.feasible:
    lay   = layout.build_layout(req, rules, floors=report.floors)
    ctrl  = control_image.render_floor_b64(lay, floor=0, mode="controlnet")
    sd    = await sd_client.generate(control_image_b64=ctrl, prompt="...")
```

## API contract (when you mount the router)

**POST `/api/design/generate`**

```json
{
  "prompt": "5 marla plot, 25x50 feet, 3 bedrooms, 2 bathrooms, kitchen, drawing, TV lounge, parking, stairs, double story.",
  "plot": { "width_ft": 25, "length_ft": 50 },   // optional; extracted from prompt otherwise
  "society": "generic",                          // generic | dha | bahria | cda
  "accept_two_story": false                      // flip to true after the user confirms two-story
}
```

Returns `DesignResponse` (see `schemas.py`):
- `feasibility.verdict` — `FEASIBLE` | `NOT_FEASIBLE`
- `layout` — structured room list (`x, y, w, h, floor, doors, windows`)
- `control_image_png_b64` — deterministic preview
- `generated_image_png_b64` — SD+ControlNet output (or control image as stub with `sd_stub_reason`)
- `explanation` — human-readable summary

**GET `/api/design/societies`** — lists configured societies.

## Environment variables (all optional)

```
GEMINI_API_KEY=…              # if unset, deterministic regex extractor is used
GEMINI_MODEL=gemini-2.5-flash
SD_CONTROLNET_URL=…           # ngrok URL from colab/sd_controlnet_server.ipynb
SD_TIMEOUT_SECONDS=180
```

## Chatbot integration

The centralized Gemini chatbot must call `POST /api/design/generate` — it is the only public surface. Feasibility cannot be bypassed. On `ONE_STORY_INSUFFICIENT`, ask the user "try two stories?" and re-submit with `accept_two_story: true`.

## Society rules

`floor_plan_2d/rules/societies/*.yaml`. `generic.yaml` is a conceptual baseline; DHA/Bahria/CDA files are `official: false` placeholders — replace with real bylaws before production. Adding a new society = one YAML file.

## Colab SD + ControlNet server

`colab/sd_controlnet_server.ipynb` — runs `runwayml/stable-diffusion-v1-5` + `lllyasviel/sd-controlnet-mlsd`, exposes `POST /generate` via pyngrok. Paste the printed URL into `SD_CONTROLNET_URL`.

## Install & test

```bash
pip install -r requirements.txt
pytest -q                    # 21 tests
```

## Files

```
floor_plan_2d/
├── __init__.py              # exports `router`
├── api.py                   # FastAPI router — /api/design/*
├── schemas.py               # pydantic models (source of truth for the layout)
├── rules/
│   ├── loader.py
│   └── societies/           # generic/dha/bahria/cda YAML
└── services/
    ├── extractor.py         # NL → Requirements (Gemini + regex fallback)
    ├── feasibility.py       # deterministic verdict, 2-story hard cap
    ├── layout.py            # grid-scan packer + validator
    ├── control_image.py     # PIL renderer (preview + controlnet modes)
    └── sd_client.py         # async HTTPX to Colab/ngrok, graceful stub

colab/
└── sd_controlnet_server.ipynb

requirements.txt
pytest.ini
```
