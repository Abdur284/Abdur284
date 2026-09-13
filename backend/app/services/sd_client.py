"""Stable Diffusion + ControlNet client.

Sends the control image (from `control_image.render_floor_b64(..., mode='controlnet')`)
plus a textual prompt to the endpoint exposed by the Colab notebook via ngrok.

Contract with the Colab server (see `colab/sd_controlnet_server.ipynb`):

  POST {SD_CONTROLNET_URL}/generate
  Content-Type: application/json
  {
    "prompt": "...",
    "negative_prompt": "...",
    "control_image_b64": "<png>",
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "controlnet_conditioning_scale": 1.1,
    "seed": 1234
  }
  -> 200 { "image_b64": "<png>" }

If `SD_CONTROLNET_URL` is empty, or the endpoint errors out, we return the
control image itself as a stub so the pipeline stays end-to-end runnable.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import httpx

from ..config import get_settings

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "top-down 2D architectural floor plan, clean black-and-white line drawing, "
    "labeled rooms, walls, doors, windows, staircase, dimensions, blueprint style, "
    "Pakistani residential house layout, high contrast, no shadows, no perspective, "
    "no furniture clutter"
)
DEFAULT_NEGATIVE = (
    "3d, perspective, isometric, photograph, render, shading, color palette, "
    "furniture in isometric view, people, text errors, blurry, watermark"
)


@dataclass
class SDResult:
    image_b64: Optional[str]
    used_sd: bool
    stub_reason: Optional[str] = None


async def generate(
    control_image_b64: str,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE,
    steps: int = 25,
    guidance: float = 7.5,
    controlnet_scale: float = 1.1,
    seed: int = 1234,
) -> SDResult:
    settings = get_settings()
    if not settings.sd_controlnet_url:
        return SDResult(image_b64=control_image_b64, used_sd=False,
                        stub_reason="SD_CONTROLNET_URL not configured — returning control image as placeholder.")

    url = settings.sd_controlnet_url.rstrip("/") + "/generate"
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "control_image_b64": control_image_b64,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "controlnet_conditioning_scale": controlnet_scale,
        "seed": seed,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.sd_timeout_seconds) as client:
            resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            return SDResult(image_b64=control_image_b64, used_sd=False,
                            stub_reason=f"SD endpoint returned HTTP {resp.status_code}.")
        data = resp.json()
        img = data.get("image_b64")
        if not img:
            return SDResult(image_b64=control_image_b64, used_sd=False,
                            stub_reason="SD endpoint response missing 'image_b64'.")
        # Sanity-check base64 payload.
        try:
            base64.b64decode(img, validate=True)
        except Exception:
            return SDResult(image_b64=control_image_b64, used_sd=False,
                            stub_reason="SD endpoint returned invalid base64.")
        return SDResult(image_b64=img, used_sd=True)
    except httpx.RequestError as e:
        log.warning("SD endpoint unreachable: %s", e)
        return SDResult(image_b64=control_image_b64, used_sd=False,
                        stub_reason=f"SD endpoint unreachable: {e.__class__.__name__}.")
    except Exception as e:  # pragma: no cover — defensive
        log.exception("SD call failed")
        return SDResult(image_b64=control_image_b64, used_sd=False,
                        stub_reason=f"SD call raised {e.__class__.__name__}.")


def build_prompt(layout, floor: int) -> Tuple[str, str]:
    """Compose an SD prompt that names the rooms visible on this floor."""
    room_bits = ", ".join(sorted({r.room_name for r in layout.rooms_on(floor)}))
    prompt = (
        f"{'ground floor' if floor == 0 else 'first floor'} 2D architectural floor plan, "
        f"top-down blueprint, rooms: {room_bits}. " + DEFAULT_PROMPT
    )
    return prompt, DEFAULT_NEGATIVE
