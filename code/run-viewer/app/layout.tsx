import type { Metadata } from "next";
import Link from "next/link";
import { Activity } from "lucide-react";
import ThemeProvider from "@/components/theme-provider";
import ThemeToggle from "@/components/theme-toggle";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

export const metadata: Metadata = {
  title: "Run viewer",
  description: "Read-only viewer for beta-training runs",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen antialiased">
        <ThemeProvider>
          <TooltipProvider>
            <header className="sticky top-0 z-20 border-b bg-background/90 backdrop-blur">
              <div className="mx-auto flex h-12 max-w-[1600px] items-center justify-between px-4">
                <Link href="/" className="flex items-center gap-2 font-semibold">
                  <Activity className="size-4" /> Run viewer
                  <span className="text-xs font-normal text-muted-foreground">read-only · beta-training/runs</span>
                </Link>
                <ThemeToggle />
              </div>
            </header>
            <main className="mx-auto flex max-w-[1600px] flex-col gap-6 px-4 py-6">{children}</main>
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
