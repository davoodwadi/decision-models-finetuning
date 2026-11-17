"use client";

import { useEffect, useState } from "react";
import { formatTime } from "@/lib/format";

// File times are formatted in the browser's time zone (the server process may resolve a different one)
export default function LocalTime({ ms }: { ms: number | null }) {
  const [text, setText] = useState("");
  useEffect(() => {
    setText(formatTime(ms));
  }, [ms]);
  return <span>{text}</span>;
}
