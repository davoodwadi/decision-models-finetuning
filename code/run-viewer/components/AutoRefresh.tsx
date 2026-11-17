"use client";

import { RefreshCw } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

// Re-render the server page every `seconds` while something on it is running
export default function AutoRefresh({ active, seconds }: { active: boolean; seconds: number }) {
  const router = useRouter();
  const [lastRefresh, setLastRefresh] = useState("");

  useEffect(() => {
    if (!active) {
      return;
    }
    const timer = setInterval(() => {
      router.refresh();
      setLastRefresh(new Date().toLocaleTimeString());
    }, seconds * 1000);
    return () => clearInterval(timer);
  }, [active, seconds, router]);

  if (!active) {
    return <span className="text-xs text-muted-foreground">not polling (nothing running)</span>;
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
      <RefreshCw className="size-3" />
      polling every {seconds} s{lastRefresh === "" ? "" : ` · last ${lastRefresh}`}
    </span>
  );
}
