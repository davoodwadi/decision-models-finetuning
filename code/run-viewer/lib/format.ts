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

// Validated categorical palette (dataviz reference palette), fixed slot order, light and dark steps.
// Slot = entity identity: callers pass a stable index (validator header order, sorted dataset, run order in the notebook).
export const SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];
export const SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"];
export const MAX_SERIES = SERIES_LIGHT.length;

export function runHref(notebook: string, run: string): string {
  return `/n/${encodeURIComponent(notebook)}/r/${encodeURIComponent(run)}`;
}

export function notebookHref(notebook: string): string {
  return `/n/${encodeURIComponent(notebook)}`;
}

// Short axis tick labels: 3 significant digits, trailing zeros dropped (0.001, 1e-4, 0.25)
export function formatTick(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
    return value.toExponential(2).replace(/\.?0+e/, "e").replace("e+", "e");
  }
  return String(Number(value.toPrecision(3)));
}
