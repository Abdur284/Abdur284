"""Render the validated `StructuredLayout` to a deterministic PNG.

The output is a clean top-view line drawing of one floor:
- black plot boundary
- dashed setback line
- solid black walls per room
- door swings as arcs
- windows as parallel double lines
- staircase as parallel treads
- parking as a hatched rectangle with a "P"
- text label per room (name + dims)

The same renderer produces both the ControlNet input (walls-only, high-contrast)
and the human-readable preview (with labels & annotations). Pixels-per-foot
scales to fit `output_px` while keeping the plot aspect ratio.
"""
from __future__ import annotations

import base64
import io
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from ..schemas import StructuredLayout, Room

Mode = Literal["controlnet", "preview"]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_room(draw: ImageDraw.ImageDraw, r: Room, ox: float, oy: float, s: float, mode: Mode) -> None:
    x0 = ox + r.x * s
    y0 = oy + r.y * s
    x1 = ox + (r.x + r.width) * s
    y1 = oy + (r.y + r.height) * s

    # Walls
    wall = 3 if mode == "controlnet" else 2
    draw.rectangle([x0, y0, x1, y1], outline="black", width=wall)

    # Parking hatch
    if r.kind == "parking":
        step = max(int(s * 1.5), 8)
        for i in range(int(x0), int(x1), step):
            draw.line([i, y0, i + (y1 - y0), y1], fill="black", width=1)

    # Staircase treads
    if r.kind == "stair":
        treads = 8
        for i in range(1, treads):
            frac = i / treads
            draw.line([x0, y0 + (y1 - y0) * frac, x1, y0 + (y1 - y0) * frac], fill="black", width=1)

    # Doors: break the wall + draw a swing arc.
    for d in r.doors:
        if d.side == "N":
            dx0 = x0 + d.offset_ft * s
            dx1 = dx0 + d.width_ft * s
            draw.line([dx0, y0, dx1, y0], fill="white", width=wall + 1)
            if mode == "preview":
                draw.arc([dx0, y0 - d.width_ft * s, dx1, y0 + d.width_ft * s], 0, 90, fill="black")
        elif d.side == "S":
            dx0 = x0 + d.offset_ft * s
            dx1 = dx0 + d.width_ft * s
            draw.line([dx0, y1, dx1, y1], fill="white", width=wall + 1)
            if mode == "preview":
                draw.arc([dx0, y1 - d.width_ft * s, dx1, y1 + d.width_ft * s], 270, 360, fill="black")
        elif d.side == "W":
            dy0 = y0 + d.offset_ft * s
            dy1 = dy0 + d.width_ft * s
            draw.line([x0, dy0, x0, dy1], fill="white", width=wall + 1)
        elif d.side == "E":
            dy0 = y0 + d.offset_ft * s
            dy1 = dy0 + d.width_ft * s
            draw.line([x1, dy0, x1, dy1], fill="white", width=wall + 1)

    # Windows: double parallel line inside the wall.
    for w in r.windows:
        if w.side == "N":
            wx0 = x0 + w.offset_ft * s
            wx1 = wx0 + w.width_ft * s
            draw.line([wx0, y0, wx1, y0], fill="white", width=wall + 1)
            draw.line([wx0, y0 - 1, wx1, y0 - 1], fill="black", width=1)
            draw.line([wx0, y0 + 2, wx1, y0 + 2], fill="black", width=1)
        elif w.side == "S":
            wx0 = x0 + w.offset_ft * s
            wx1 = wx0 + w.width_ft * s
            draw.line([wx0, y1, wx1, y1], fill="white", width=wall + 1)
            draw.line([wx0, y1 - 2, wx1, y1 - 2], fill="black", width=1)
            draw.line([wx0, y1 + 1, wx1, y1 + 1], fill="black", width=1)
        elif w.side == "W":
            wy0 = y0 + w.offset_ft * s
            wy1 = wy0 + w.width_ft * s
            draw.line([x0, wy0, x0, wy1], fill="white", width=wall + 1)
            draw.line([x0 - 1, wy0, x0 - 1, wy1], fill="black", width=1)
            draw.line([x0 + 2, wy0, x0 + 2, wy1], fill="black", width=1)
        elif w.side == "E":
            wy0 = y0 + w.offset_ft * s
            wy1 = wy0 + w.width_ft * s
            draw.line([x1, wy0, x1, wy1], fill="white", width=wall + 1)
            draw.line([x1 - 2, wy0, x1 - 2, wy1], fill="black", width=1)
            draw.line([x1 + 1, wy0, x1 + 1, wy1], fill="black", width=1)

    # Labels (preview only)
    if mode == "preview":
        font = _font(max(int(s * 1.2), 10))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        label = r.room_name
        dims = f"{r.width:.0f}'x{r.height:.0f}'"
        try:
            bbox = font.getbbox(label)
            w_text = bbox[2] - bbox[0]
            h_text = bbox[3] - bbox[1]
        except AttributeError:  # pragma: no cover — older Pillow
            w_text, h_text = font.getsize(label)
        draw.text((cx - w_text / 2, cy - h_text), label, fill="black", font=font)
        draw.text((cx - w_text / 2, cy + 2), dims, fill="#444", font=_font(max(int(s * 0.9), 8)))


def render_floor_png(layout: StructuredLayout, floor: int, mode: Mode = "preview",
                     output_px: int = 768, margin_px: int = 40) -> bytes:
    """Return PNG bytes for the given floor. `mode='controlnet'` = walls-only."""
    plot_w, plot_h = layout.plot.width_ft, layout.plot.length_ft
    aspect = plot_w / plot_h
    # Fit within a square canvas.
    if aspect >= 1:
        canvas_w = output_px
        canvas_h = int(output_px / aspect)
    else:
        canvas_h = output_px
        canvas_w = int(output_px * aspect)
    canvas_w = max(canvas_w, 200)
    canvas_h = max(canvas_h, 200)

    img = Image.new("RGB", (canvas_w + 2 * margin_px, canvas_h + 2 * margin_px), "white")
    draw = ImageDraw.Draw(img)

    s = min(canvas_w / plot_w, canvas_h / plot_h)
    ox = margin_px
    oy = margin_px

    # Plot boundary (thick black)
    draw.rectangle([ox, oy, ox + plot_w * s, oy + plot_h * s], outline="black", width=4)

    # Setback (dashed)
    sx0 = ox + layout.buildable_x * s
    sy0 = oy + layout.buildable_y * s
    sx1 = ox + (layout.buildable_x + layout.buildable_w) * s
    sy1 = oy + (layout.buildable_y + layout.buildable_h) * s
    if mode == "preview":
        for i in range(int(sx0), int(sx1), 10):
            draw.line([i, sy0, i + 5, sy0], fill="#888", width=1)
            draw.line([i, sy1, i + 5, sy1], fill="#888", width=1)
        for j in range(int(sy0), int(sy1), 10):
            draw.line([sx0, j, sx0, j + 5], fill="#888", width=1)
            draw.line([sx1, j, sx1, j + 5], fill="#888", width=1)

    for r in layout.rooms_on(floor):
        _draw_room(draw, r, ox, oy, s, mode)

    if mode == "preview":
        title = f"{'GROUND' if floor == 0 else 'FIRST'} FLOOR — {plot_w:.0f}' x {plot_h:.0f}'"
        draw.text((ox, 4), title, fill="black", font=_font(14))

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_floor_b64(layout: StructuredLayout, floor: int, mode: Mode = "preview") -> str:
    return base64.b64encode(render_floor_png(layout, floor, mode)).decode("ascii")
