import type { ReactNode } from "react";

export const metadata = {
  title: "HAVEN AI — 2D Floor Plan Design",
  description: "Feasibility-checked 2D floor-plan generator using Stable Diffusion + ControlNet.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body style={{
        fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        margin: 0, background: "#0b0d10", color: "#e6e9ee", minHeight: "100vh",
      }}>
        {children}
      </body>
    </html>
  );
}
