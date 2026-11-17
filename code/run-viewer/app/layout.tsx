import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Run viewer",
  description: "Read-only viewer for beta-training runs",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="top">
          <Link href="/">Run viewer</Link> <span className="muted small">read-only · beta-training/runs</span>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
