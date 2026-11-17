import "server-only";
import fs from "node:fs";
import path from "node:path";
import { Cell, Table, columnIndex, numberAt, parseCsv } from "./csv";

// Read-only access to code/train-notebooks/beta-training/runs/<notebook>/<run>/.
// Only run.json, steps.csv and val_history.csv are read. checkpoints/ and betas/ are never listed or opened.

const STATUSES = ["running", "interrupted", "finished"] as const;
export type Status = (typeof STATUSES)[number];

export function runsRoot(): string {
  const configured = process.env.RUNS_DIR;
  const root =
    configured === undefined
      ? path.resolve(/*turbopackIgnore: true*/ process.cwd(), "..", "train-notebooks", "beta-training", "runs")
      : path.resolve(configured);
  if (!fs.existsSync(root)) {
    throw new Error(`Runs folder ${root} does not exist (set RUNS_DIR to override)`);
  }
  return root;
}

function subdirectories(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const names: string[] = [];
  for (const entry of entries) {
    if (entry.isDirectory()) {
      names.push(entry.name);
    }
  }
  names.sort();
  return names;
}

function requireName(name: string, options: string[], kind: string): string {
  if (!options.includes(name)) {
    throw new Error(`No ${kind} named "${name}"`);
  }
  return name;
}

// ---------------------------------------------------------------- notebooks

export type NotebookInfo = {
  name: string;
  runCount: number;
  otherFolders: string[];
  newestUpdate: number | null;
};

export function listNotebooks(): NotebookInfo[] {
  const root = runsRoot();
  const notebooks: NotebookInfo[] = [];
  for (const name of subdirectories(root)) {
    const { runs, otherFolders } = splitRunFolders(path.join(root, name));
    let newest: number | null = null;
    for (const run of runs) {
      const updated = runUpdatedTime(path.join(root, name, run));
      if (newest === null || updated > newest) {
        newest = updated;
      }
    }
    notebooks.push({ name, runCount: runs.length, otherFolders, newestUpdate: newest });
  }
  notebooks.sort((a, b) => (b.newestUpdate ?? 0) - (a.newestUpdate ?? 0));
  return notebooks;
}

function splitRunFolders(notebookDir: string): { runs: string[]; otherFolders: string[] } {
  // A run folder is one holding run.json; anything else (comparison tables, step-0 validation) is listed apart
  const runs: string[] = [];
  const otherFolders: string[] = [];
  for (const name of subdirectories(notebookDir)) {
    if (fs.existsSync(path.join(notebookDir, name, "run.json"))) {
      runs.push(name);
    } else {
      otherFolders.push(name);
    }
  }
  return { runs, otherFolders };
}

function notebookDir(notebook: string): string {
  const root = runsRoot();
  requireName(notebook, subdirectories(root), "notebook");
  return path.join(root, notebook);
}

function runDir(notebook: string, run: string): string {
  const dir = notebookDir(notebook);
  requireName(run, splitRunFolders(dir).runs, `run in ${notebook}`);
  return path.join(dir, run);
}

function runUpdatedTime(dir: string): number {
  let newest = 0;
  for (const file of ["run.json", "steps.csv", "val_history.csv"]) {
    const full = path.join(/*turbopackIgnore: true*/ dir, file);
    if (fs.existsSync(full)) {
      newest = Math.max(newest, fs.statSync(full).mtimeMs);
    }
  }
  return newest;
}

// ---------------------------------------------------------------- run.json

type JsonObject = Record<string, unknown>;

function requireKey(obj: JsonObject, key: string, where: string): unknown {
  if (!(key in obj)) {
    throw new Error(`${where}: missing key "${key}"`);
  }
  return obj[key];
}

function requireObject(value: unknown, where: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${where}: expected an object`);
  }
  return value as JsonObject;
}

function requireString(value: unknown, where: string): string {
  if (typeof value !== "string") {
    throw new Error(`${where}: expected a string, got ${JSON.stringify(value)}`);
  }
  return value;
}

function requireNumber(value: unknown, where: string): number {
  if (typeof value !== "number") {
    throw new Error(`${where}: expected a number, got ${JSON.stringify(value)}`);
  }
  return value;
}

type RunJson = {
  status: Status;
  created: string;
  config: JsonObject;
  model: JsonObject | null;
  bestMetric: string;
  bestMode: "min" | "max";
  resumes: unknown[];
  summary: JsonObject | null;
};

function readRunJson(dir: string): RunJson {
  const file = path.join(dir, "run.json");
  const raw = fs.readFileSync(file, "utf8");
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`${file}: invalid JSON (${(error as Error).message})`);
  }
  const obj = requireObject(parsed, file);
  if (!("identity" in obj)) {
    throw new Error(`${file}: no "identity" key; this is an older run.json format the viewer does not read`);
  }
  const identity = requireObject(obj.identity, `${file} identity`);
  const status = requireString(requireKey(obj, "status", file), `${file} status`);
  if (!(STATUSES as readonly string[]).includes(status)) {
    throw new Error(`${file}: unknown status "${status}"`);
  }
  const bestMode = requireString(requireKey(identity, "best_mode", `${file} identity`), `${file} best_mode`);
  if (bestMode !== "min" && bestMode !== "max") {
    throw new Error(`${file}: best_mode must be min or max, got "${bestMode}"`);
  }
  const summaryValue = requireKey(obj, "summary", file);
  const runMetadata = requireObject(requireKey(identity, "run_metadata", `${file} identity`), `${file} run_metadata`);
  const resumes = requireKey(obj, "resumes", file);
  if (!Array.isArray(resumes)) {
    throw new Error(`${file}: resumes must be a list`);
  }
  return {
    status: status as Status,
    created: requireString(requireKey(obj, "created", file), `${file} created`),
    config: requireObject(requireKey(identity, "config", `${file} identity`), `${file} config`),
    model: "model" in runMetadata ? requireObject(runMetadata.model, `${file} run_metadata.model`) : null,
    bestMetric: requireString(requireKey(identity, "best_metric", `${file} identity`), `${file} best_metric`),
    bestMode,
    resumes,
    summary: summaryValue === null ? null : requireObject(summaryValue, `${file} summary`),
  };
}

// ---------------------------------------------------------------- CSV files

function readTable(dir: string, file: string, status: Status): Table {
  const full = path.join(/*turbopackIgnore: true*/ dir, file);
  if (!fs.existsSync(full)) {
    // A run that has just started has written run.json but no tables yet
    if (status === "running") {
      return { columns: [], rows: [], droppedPartialLine: false };
    }
    throw new Error(`${full} is missing (run status is ${status})`);
  }
  return parseCsv(fs.readFileSync(full, "utf8"), full);
}

// ---------------------------------------------------------------- run summary

export type RunSummary = {
  name: string;
  status: Status;
  created: string;
  stepsDone: number;
  totalSteps: number | null;
  bestMetric: string;
  bestMode: "min" | "max";
  bestStep: number | null;
  bestValue: number | null;
  // true when best step/value come from the finished run's summary, false when computed from val_history so far
  bestFromSummary: boolean;
  updated: number;
  config: JsonObject;
  model: JsonObject | null;
  resumes: unknown[];
};

export type RunRow = { name: string; summary: RunSummary } | { name: string; error: string };

function lastStep(steps: Table, file: string): number {
  if (steps.rows.length === 0) {
    return 0;
  }
  const index = columnIndex(steps, "step", file);
  const value = numberAt(steps.rows[steps.rows.length - 1], index, "step", file);
  if (value === null) {
    throw new Error(`${file}: last row has no step`);
  }
  return value;
}

function bestFromHistory(val: Table, metric: string, mode: "min" | "max", file: string): { step: number; value: number } | null {
  if (val.rows.length === 0) {
    return null;
  }
  const stepIndex = columnIndex(val, "step", file);
  const metricIndex = columnIndex(val, metric, file);
  let best: { step: number; value: number } | null = null;
  for (const row of val.rows) {
    const value = numberAt(row, metricIndex, metric, file);
    if (value === null) {
      continue;
    }
    const step = numberAt(row, stepIndex, "step", file);
    if (step === null) {
      throw new Error(`${file}: a row has no step`);
    }
    const better = best === null || (mode === "min" ? value < best.value : value > best.value);
    if (better) {
      best = { step, value };
    }
  }
  return best;
}

function summarize(name: string, dir: string, info: RunJson, steps: Table, val: Table): RunSummary {
  const stepsFile = path.join(dir, "steps.csv");
  const valFile = path.join(dir, "val_history.csv");
  let totalSteps: number | null = null;
  let bestStep: number | null = null;
  let bestValue: number | null = null;
  let bestFromSummary = false;
  if (info.summary !== null && "total_steps" in info.summary) {
    // A finished run's summary is authoritative
    const where = `${dir}/run.json summary`;
    totalSteps = requireNumber(info.summary.total_steps, `${where}.total_steps`);
    bestStep = requireNumber(requireKey(info.summary, "best_step", where), `${where}.best_step`);
    bestValue = requireNumber(requireKey(info.summary, "best_value", where), `${where}.best_value`);
    bestFromSummary = true;
  } else {
    const best = bestFromHistory(val, info.bestMetric, info.bestMode, valFile);
    if (best !== null) {
      bestStep = best.step;
      bestValue = best.value;
    }
  }
  return {
    name,
    status: info.status,
    created: info.created,
    stepsDone: lastStep(steps, stepsFile),
    totalSteps,
    bestMetric: info.bestMetric,
    bestMode: info.bestMode,
    bestStep,
    bestValue,
    bestFromSummary,
    updated: runUpdatedTime(dir),
    config: info.config,
    model: info.model,
    resumes: info.resumes,
  };
}

export type NotebookRuns = { notebook: string; runs: RunRow[]; otherFolders: string[] };

export function listRuns(notebook: string): NotebookRuns {
  const dir = notebookDir(notebook);
  const { runs, otherFolders } = splitRunFolders(dir);
  const rows: RunRow[] = [];
  for (const name of runs) {
    const runPath = path.join(dir, name);
    // One broken run must not hide the others: its error is shown in its own row
    try {
      const info = readRunJson(runPath);
      const steps = readTable(runPath, "steps.csv", info.status);
      const val = readTable(runPath, "val_history.csv", info.status);
      rows.push({ name, summary: summarize(name, runPath, info, steps, val) });
    } catch (error) {
      rows.push({ name, error: (error as Error).message });
    }
  }
  return { notebook, runs: rows, otherFolders };
}

// ---------------------------------------------------------------- run detail

export type RunDetail = {
  notebook: string;
  summary: RunSummary;
  val: Table;
  steps: Table;
};

export function readRun(notebook: string, run: string): RunDetail {
  const dir = runDir(notebook, run);
  const info = readRunJson(dir);
  const steps = readTable(dir, "steps.csv", info.status);
  const val = readTable(dir, "val_history.csv", info.status);
  return { notebook, summary: summarize(run, dir, info, steps, val), val, steps };
}

// ---------------------------------------------------------------- column helpers

// Validation columns are named <validator>/<metric>; everything without a "/" is run-level (step, lr, train_loss, ...)
export function validatorMetrics(columns: string[]): { validators: string[]; metrics: string[] } {
  const validators: string[] = [];
  const metrics: string[] = [];
  for (const column of columns) {
    const slash = column.indexOf("/");
    if (slash === -1) {
      continue;
    }
    const validator = column.slice(0, slash);
    const metric = column.slice(slash + 1);
    if (!validators.includes(validator)) {
      validators.push(validator);
    }
    if (!metrics.includes(metric)) {
      metrics.push(metric);
    }
  }
  return { validators, metrics };
}

export type { Cell, Table };
