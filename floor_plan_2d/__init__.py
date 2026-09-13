"""HAVEN AI — 2D Floor Plan Design module.

Drop this package into your existing FastAPI backend and mount `router`:

    from fastapi import FastAPI
    from floor_plan_2d import router as floor_plan_router

    app = FastAPI()
    app.include_router(floor_plan_router)

Or call the services directly (bypassing FastAPI):

    from floor_plan_2d.services import extractor, feasibility, layout, control_image, sd_client
    from floor_plan_2d.rules.loader import load_rules

Environment variables the module reads (all optional):

    GEMINI_API_KEY              # if set, uses Gemini for NL parsing; else regex fallback
    GEMINI_MODEL                # default: gemini-2.5-flash
    SD_CONTROLNET_URL           # ngrok URL from colab/sd_controlnet_server.ipynb
    SD_TIMEOUT_SECONDS          # default: 180
"""
from .api import router

__all__ = ["router"]
