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

export function notebookHref(source: string, notebook: string): string {
  return `/s/${encodeURIComponent(source)}/n/${encodeURIComponent(notebook)}`;
}

export function runHref(source: string, notebook: string, run: string): string {
  return `${notebookHref(source, notebook)}/r/${encodeURIComponent(run)}`;
}

// Home page section of one run source
export function sourceHref(source: string): string {
  return `/#source-${encodeURIComponent(source)}`;
}

export function compareHref(source: string, notebook: string, query: string): string {
  return `${notebookHref(source, notebook)}/compare?${query}`;
}

// Short axis tick labels: 3 significant digits, trailing zeros dropped (0.001, 1e-4, 0.25)
export function formatTick(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) {
    return value.toExponential(2).replace(/\.?0+e/, "e").replace("e+", "e");
  }
  return String(Number(value.toPrecision(3)));
}

// Main hyperparameters shown as runs-table columns, in this order, when at least one run of the notebook has the key
// in its run.json config. The run page's Config tab lists every key.
export const MAIN_HYPERPARAMETERS: { key: string; label: string }[] = [
  { key: "ce_datasets", label: "CE data" },
  { key: "beta_datasets", label: "table data" },
  { key: "beta_lambda", label: "λ" },
  { key: "kl_anchor_alpha", label: "α" },
  { key: "lr", label: "lr" },
  { key: "num_epochs", label: "epochs" },
  { key: "micro_chunk", label: "micro-chunk" },
];

// One config value as a table cell: numbers stay numbers (sortable), lists become "a, b" ("—" when empty)
function hyperparameterCell(value: unknown): string | number {
  if (typeof value === "number" || typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.length === 0 ? "—" : value.join(", ");
  }
  return JSON.stringify(value);
}

// The MAIN_HYPERPARAMETERS a run's config holds, keyed by config key (a key is absent when the config lacks it)
export function mainHyperparameters(config: Record<string, unknown>): Record<string, string | number> {
  const values: Record<string, string | number> = {};
  for (const { key } of MAIN_HYPERPARAMETERS) {
    if (key in config) {
      values[key] = hyperparameterCell(config[key]);
    }
  }
  return values;
}

// The MAIN_HYPERPARAMETERS that at least one of these runs has: the columns to show
export function hyperparameterColumns(runs: { hyperparameters: Record<string, string | number> }[]): { key: string; label: string }[] {
  return MAIN_HYPERPARAMETERS.filter(({ key }) => runs.some((run) => key in run.hyperparameters));
}
