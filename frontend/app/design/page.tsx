"use client";

import { useState } from "react";
import { generateDesign, DesignResponse } from "../../lib/api";

const EXAMPLES = [
  "I have a 5 marla plot 25 x 50 feet. I want 3 bedrooms, 2 bathrooms, one kitchen, a drawing room, TV lounge, car parking and stairs. I want a double-story house.",
  "10 marla plot 35 x 65 feet. 4 bedrooms with attached baths, kitchen, drawing room, TV lounge, guest room, parking.",
  "25x50 feet, 6 bedrooms, 6 bathrooms, kitchen, drawing, TV lounge, parking.",
];

export default function DesignPage() {
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [society, setSociety] = useState("generic");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resp, setResp] = useState<DesignResponse | null>(null);
  const [acceptTwoStory, setAcceptTwoStory] = useState(false);

  async function run(accept: boolean = acceptTwoStory) {
    setLoading(true);
    setError(null);
    try {
      const r = await generateDesign({ prompt, society, accept_two_story: accept });
      setResp(r);
      setAcceptTwoStory(accept);
    } catch (e: any) {
      setError(e.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const needsTwoStoryConfirm =
    resp && !resp.feasibility.feasible &&
    resp.feasibility.issues.some(i => i.code === "ONE_STORY_INSUFFICIENT");

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px" }}>
      <h1 style={{ marginTop: 0 }}>2D Floor Plan Designer</h1>

      <label style={{ display: "block", color: "#9aa3af", fontSize: 14, marginBottom: 4 }}>Prompt</label>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={5}
        style={{
          width: "100%", padding: 12, borderRadius: 8, border: "1px solid #2a2f36",
          background: "#0f1216", color: "#e6e9ee", fontFamily: "inherit", fontSize: 14,
        }}
      />
      <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
        {EXAMPLES.map((ex, i) => (
          <button key={i} onClick={() => setPrompt(ex)} style={btnSecondary}>Example {i + 1}</button>
        ))}
      </div>

      <div style={{ display: "flex", gap: 12, marginTop: 16, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ color: "#9aa3af", fontSize: 14 }}>Society:</label>
        <select value={society} onChange={(e) => setSociety(e.target.value)} style={select}>
          <option value="generic">Generic Pakistan Residential</option>
          <option value="dha">DHA (conceptual)</option>
          <option value="bahria">Bahria Town (conceptual)</option>
          <option value="cda">CDA / Islamabad (conceptual)</option>
        </select>
        <button onClick={() => run(false)} disabled={loading} style={btnPrimary}>
          {loading ? "Working…" : "Generate 2D Floor Plan"}
        </button>
      </div>

      {error && <div style={errBox}>{error}</div>}

      {resp && (
        <section style={{ marginTop: 32, display: "grid", gap: 24 }}>
          <div style={card}>
            <h2 style={{ marginTop: 0 }}>Feasibility: {resp.feasibility.verdict}</h2>
            <p style={{ color: "#cbd5e1" }}>{resp.explanation}</p>
            {resp.feasibility.issues.length > 0 && (
              <ul>{resp.feasibility.issues.map((i, idx) => (
                <li key={idx}><b>{i.code}</b> — {i.message}{i.suggestion ? ` (${i.suggestion})` : ""}</li>
              ))}</ul>
            )}
            {resp.feasibility.notes.map((n, i) => <p key={i} style={{ color: "#94a3b8" }}>{n}</p>)}
            {needsTwoStoryConfirm && (
              <button onClick={() => run(true)} style={btnPrimary}>Yes — try a two-story design</button>
            )}
          </div>

          {resp.layout && (
            <div style={card}>
              <h2 style={{ marginTop: 0 }}>Proposed Design</h2>
              {Array.from({ length: resp.layout.floors }).map((_, f) => (
                <div key={f} style={{ marginBottom: 12 }}>
                  <h3 style={{ margin: "8px 0" }}>{f === 0 ? "Ground Floor" : "First Floor"}</h3>
                  <ul style={{ margin: 0 }}>
                    {resp.layout!.rooms.filter(r => r.floor === f).map((r, i) => (
                      <li key={i}>{r.room_name} — {r.width.toFixed(0)}′ × {r.height.toFixed(0)}′</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}

          {resp.control_image_png_b64 && (
            <div style={card}>
              <h2 style={{ marginTop: 0 }}>Structured Layout (deterministic)</h2>
              <img alt="structured layout" src={`data:image/png;base64,${resp.control_image_png_b64}`}
                   style={{ width: "100%", background: "white", borderRadius: 6 }} />
            </div>
          )}

          {resp.generated_image_png_b64 && (
            <div style={card}>
              <h2 style={{ marginTop: 0 }}>
                Generated 2D Floor Plan {resp.sd_used ? "(Stable Diffusion + ControlNet)" : "(Stub)"}
              </h2>
              {resp.sd_stub_reason && (
                <p style={{ color: "#eab308", fontSize: 13 }}>{resp.sd_stub_reason}</p>
              )}
              <img alt="generated plan" src={`data:image/png;base64,${resp.generated_image_png_b64}`}
                   style={{ width: "100%", background: "white", borderRadius: 6 }} />
            </div>
          )}
        </section>
      )}
    </main>
  );
}

const btnPrimary: React.CSSProperties = {
  padding: "10px 16px", background: "#3b82f6", color: "white", border: "none",
  borderRadius: 8, cursor: "pointer", fontSize: 14, fontWeight: 600,
};
const btnSecondary: React.CSSProperties = {
  padding: "6px 12px", background: "#1f2937", color: "#e6e9ee", border: "1px solid #374151",
  borderRadius: 6, cursor: "pointer", fontSize: 12,
};
const select: React.CSSProperties = {
  padding: "8px 12px", background: "#0f1216", color: "#e6e9ee", border: "1px solid #2a2f36",
  borderRadius: 6, fontSize: 14,
};
const card: React.CSSProperties = {
  background: "#111418", border: "1px solid #1f2937", borderRadius: 10, padding: 20,
};
const errBox: React.CSSProperties = {
  marginTop: 16, padding: 12, background: "#3f1d1d", border: "1px solid #7f1d1d",
  borderRadius: 8, color: "#fecaca",
};
