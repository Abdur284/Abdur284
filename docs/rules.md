# Society Rules

Rules live in `backend/app/rules/societies/*.yaml`. The loader accepts these keys via `society` in the API request:

- `generic` (default) — Generic Pakistan Residential (conceptual defaults)
- `dha` — DHA (conceptual placeholder; replace with real bylaws before production)
- `bahria` — Bahria Town (conceptual placeholder)
- `cda` — CDA / Islamabad (conceptual placeholder)

Only `generic` is a considered conceptual baseline; the others are placeholders with `official: false` and must be updated with the actual society bylaws before real use.

## Schema

```yaml
name: "..."
official: false           # true only if you've verified with the source authority
notes: [...]

setbacks:                 # feet stripped off each side; 0 = build to the boundary
  front_ft: 5
  rear_ft: 0
  side_ft: 0

max_coverage_pct: 85      # of total plot area
max_floors: 2             # HARD CAP — the pipeline never exceeds 2

min_rooms:
  bedroom:  { w: 10, h: 10, area: 110 }
  master:   { w: 12, h: 13, area: 160 }
  bathroom: { w: 5,  h: 7,  area: 35  }
  kitchen:  { w: 8,  h: 10, area: 80  }
  drawing:  { w: 12, h: 13, area: 160 }
  tv_lounge:{ w: 12, h: 12, area: 144 }
  guest:    { w: 10, h: 11, area: 110 }
  servant:  { w: 8,  h: 9,  area: 72  }
  stair:    { w: 8,  h: 10, area: 80  }
  parking:  { w: 9,  h: 18, area: 162 }
  corridor: { w: 3,  h: 3,  area: 9   }

parking:
  required_when_area_sqft_gte: 900
  min_cars: 1

circulation_pct: 12       # added to interior room area for feasibility budget
```

## Adding a new society

1. Copy `generic.yaml` to `<name>.yaml`.
2. Adjust setbacks, coverage, min dimensions, parking rules.
3. If the values come from an official source, set `official: true` and cite it in `notes`.
4. `GET /api/design/societies` will pick up the file automatically.
