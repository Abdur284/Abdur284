# HAVEN AI — 2D Floor Plan Design Module

Standalone service for HAVEN AI's 2D House / Floor Plan Design module.

The user gives a plot (marla or feet) and a natural‑language brief. The service:

1. Extracts a structured requirement (Gemini if `GEMINI_API_KEY` is set, deterministic regex fallback otherwise).
2. Runs a **deterministic feasibility engine** against configurable Pakistani society rules (Generic / DHA / Bahria / CDA / Custom).
3. If not feasible on 1 story, proposes a 2‑story layout (never more than 2). If still infeasible, returns `NOT_FEASIBLE` with reasons and a suggested modification.
4. Builds a **validated structured layout** (rooms with `x, y, w, h, doors, windows, floor`).
5. Renders a deterministic **ControlNet input image** (walls, doors, windows, stairs, parking, plot boundary).
6. Calls a **Stable Diffusion + ControlNet** endpoint (Colab + ngrok in dev) to produce the visual 2D floor plan.
7. Returns feasibility verdict, per‑floor room list, structured layout JSON, control image and generated image.

The structured layout is the **source of truth**. Stable Diffusion + ControlNet only produce the visual.

## Layout

```
backend/       FastAPI service (feasibility, layout, control image, SD client, API)
frontend/      Next.js 14 app (prompt UI, feasibility card, generated plan viewer)
colab/         Reference SD + ControlNet server for Google Colab + ngrok
docs/          PRD, architecture, rules, design notes
```

## Quick start

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in later when you have keys
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev             # http://localhost:3000
```

### Colab (SD + ControlNet)

Open `colab/sd_controlnet_server.ipynb` in Google Colab, run all cells, copy the printed ngrok URL, and paste it into `backend/.env` as `SD_CONTROLNET_URL=...`. Restart the backend.

## Chatbot integration

The centralized Gemini chatbot must POST to `/api/design/generate` with the user's prompt. It **must not** bypass the feasibility engine. See `docs/architecture.md`.

## Tests

```bash
cd backend
pytest -q
```

Covers: 5 marla / 10 marla / custom / single- vs two-story / too many rooms / invalid & missing dims / conflicting requirements / parking / staircase / SD failure / ControlNet failure / ambiguous prompt / overlap detection / out-of-bounds detection.
