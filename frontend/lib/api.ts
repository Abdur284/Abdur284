// Thin client for the FastAPI backend. All calls go through this file so the
// rest of the UI stays framework-agnostic.

const BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type PlotDimensions = { width_ft: number; length_ft: number };

export type Room = {
  room_name: string;
  kind: string;
  floor: number;
  x: number; y: number; width: number; height: number;
  doors: { side: "N" | "S" | "E" | "W"; offset_ft: number; width_ft: number }[];
  windows: { side: "N" | "S" | "E" | "W"; offset_ft: number; width_ft: number }[];
};

export type StructuredLayout = {
  plot: PlotDimensions;
  setback_front_ft: number; setback_rear_ft: number; setback_side_ft: number;
  buildable_x: number; buildable_y: number; buildable_w: number; buildable_h: number;
  floors: number;
  rooms: Room[];
};

export type FeasibilityIssue = { code: string; message: string; suggestion?: string };
export type FeasibilityReport = {
  feasible: boolean;
  verdict: "FEASIBLE" | "NOT_FEASIBLE";
  floors: number;
  issues: FeasibilityIssue[];
  notes: string[];
  suggested_alternative?: string;
};

export type DesignResponse = {
  requirements: any;
  feasibility: FeasibilityReport;
  layout?: StructuredLayout;
  control_image_png_b64?: string;
  generated_image_png_b64?: string;
  sd_used: boolean;
  sd_stub_reason?: string;
  explanation: string;
};

export async function generateDesign(body: {
  prompt: string;
  plot?: PlotDimensions;
  society?: string;
  accept_two_story?: boolean;
}): Promise<DesignResponse> {
  const res = await fetch(`${BASE}/api/design/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
