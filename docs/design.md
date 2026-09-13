# Design Notes

## Why the layout is deterministic, not learned

Stable Diffusion is a generative image model, not an architectural constraint solver. Letting it decide room positions produces plausible-looking blueprints that ignore setbacks, overlap rooms, and hallucinate floors. So:

- **Numbers, geometry, and feasibility** are decided by deterministic Python.
- **Only the visual rendering** is delegated to SD + ControlNet.
- The **ControlNet input image** encodes the deterministic geometry (walls, doors, windows, plot boundary, stairs, parking) so the generated image tracks the source of truth.

## Layout algorithm

`_pack()` in `backend/app/services/layout.py`:

1. Sort rooms by area, largest first.
2. For each room, try both orientations (w×h and h×w).
3. Grid-scan the buildable rectangle in 1-ft steps; place at the top-leftmost slot that doesn't collide with any already-placed room *or* pre-reserved rectangle.
4. Reserve the staircase at a fixed corner (top-right of the buildable rect) so its footprint is identical on ground and first floor — required for a physically valid 2-story stair well.

After placement, `validate()` checks:
- All rooms lie inside the buildable rectangle.
- No two rooms overlap.
- On 2-story: the stair exists on both floors and its rectangle matches.

If validation fails, the API returns `LAYOUT_FAILED` rather than pretending it worked.

## Doors and windows

Doors face the *interior* (an edge not on the plot boundary). Windows face the *exterior* (an edge that touches the plot boundary). Bathrooms, staircase, and parking skip windows. On small plots where every edge of a room touches the boundary, the door falls back to south — flagged in the visual, kept deterministic.

## Marla conversion

We use **1 marla = 225 sqft** (Punjab urban). If the user gives marla without dimensions, we use a preset table (3/5/7/8/10/12/14/20 marla) and fall back to a square-ish plot otherwise. The user can always override with `plot.width_ft / plot.length_ft`.

## Prompt engineering for SD

Positive prompt names the visible rooms on the floor and enforces "top-down 2D architectural floor plan, blueprint style". Negative prompt kills 3D/isometric/photograph interpretations. The `controlnet_conditioning_scale` defaults to `1.1` — biased toward following the control image over the prompt.
