export function formatNumber(value: number | null): string {
  if (value === null) {
    return "";
  }
  if (Number.isInteger(value) && Math.abs(value) < 1e7) {
    return String(value);
  }
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
    return value.toExponential(3);
  }
  return value.toPrecision(4);
}

export function formatTime(ms: number | null): string {
  if (ms === null) {
    return "—";
  }
  const date = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function formatAge(ms: number | null, now: number): string {
  if (ms === null) {
    return "";
  }
  const seconds = Math.round((now - ms) / 1000);
  if (seconds < 90) {
    return `${seconds}s ago`;
  }
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) {
    return `${minutes} min ago`;
  }
  const hours = Math.round(minutes / 60);
  if (hours < 36) {
    return `${hours} h ago`;
  }
  return `${Math.round(hours / 24)} days ago`;
}

export const SERIES_COLORS = [
  "#2563eb",
  "#dc2626",
  "#16a34a",
  "#d97706",
  "#9333ea",
  "#0891b2",
  "#db2777",
  "#65a30d",
  "#475569",
  "#ea580c",
];

export function runHref(notebook: string, run: string): string {
  return `/n/${encodeURIComponent(notebook)}/r/${encodeURIComponent(run)}`;
}

export function notebookHref(notebook: string): string {
  return `/n/${encodeURIComponent(notebook)}`;
}
