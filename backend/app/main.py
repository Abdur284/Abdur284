"""FastAPI entrypoint for HAVEN AI — 2D Floor Plan Design module."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.design import router as design_router
from .config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="HAVEN AI — 2D Floor Plan Design",
    version="0.1.0",
    description="Feasibility-checked, Pakistan-oriented 2D floor-plan generator using Stable Diffusion + ControlNet.",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(design_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(settings.gemini_api_key),
        "sd_configured": bool(settings.sd_controlnet_url),
    }
