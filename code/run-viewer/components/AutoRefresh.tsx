"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

// Re-render the server page every `seconds` while the run is running
export default function AutoRefresh({ active, seconds }: { active: boolean; seconds: number }) {
  const router = useRouter();
  const [lastRefresh, setLastRefresh] = useState<string>("");

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
    return <span className="muted">not polling (nothing running)</span>;
  }
  return (
    <span className="muted">
      polling every {seconds} s{lastRefresh === "" ? "" : `, last refresh ${lastRefresh}`}
    </span>
  );
}
