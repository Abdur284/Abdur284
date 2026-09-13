import Link from "next/link";

export default function Home() {
  return (
    <main style={{ maxWidth: 880, margin: "0 auto", padding: "48px 24px" }}>
      <h1 style={{ margin: 0, fontSize: 32 }}>HAVEN AI</h1>
      <p style={{ color: "#9aa3af", marginTop: 8 }}>
        2D House / Floor Plan Design module. Enter your plot and requirements — the system runs a
        feasibility check first, then generates the plan with Stable Diffusion + ControlNet.
      </p>
      <Link href="/design" style={{
        display: "inline-block", marginTop: 24, padding: "12px 20px",
        background: "#3b82f6", color: "white", borderRadius: 8, textDecoration: "none",
      }}>Open 2D Designer →</Link>
    </main>
  );
}
