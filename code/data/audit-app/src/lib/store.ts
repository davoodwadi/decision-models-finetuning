import { execFile } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { parseCsv, stringifyCsv } from "./csv";
import {
  classifyAuditRow,
  makeRowKey,
  normalizeQuestionOverride,
  normalizeTagList,
  resolveContextText,
} from "./triage";
import type {
  AuditRow,
  ContextCandidate,
  DatasetInfo,
  DatasetPayload,
  ItemDetail,
  QuestionOverride,
  SchemaKind,
} from "./types";

const execFileAsync = promisify(execFile);

const APP_DIR = process.cwd();
const DATA_DIR = path.resolve(APP_DIR, "..");
const REPO_ROOT = path.resolve(DATA_DIR, "..", "..");
const PYTHON_BIN = path.join(REPO_ROOT, ".venv-cu13", "bin", "python");
const BUILD_INDEX_SCRIPT = path.join(APP_DIR, "scripts", "build_index.py");
const DETAILS_FILE = ".audit_details.json";
const OVERRIDES_FILE = "question_overrides.json";
const TAGS_FILE = "question_tags.json";

interface RawDetailsBundle {
  version: number;
  schema_kind: SchemaKind;
  study_contexts: Record<string, ContextCandidate[]>;
  items: Record<string, ItemDetail>;
}

interface CachedDataset {
  dsId: string;
  csvPath: string;
  datasetDir: string;
  progressPath: string;
  detailsPath: string;
  overridesPath: string;
  tagsPath: string;
  hasNotebook: boolean;
  schemaKind: SchemaKind;
  columns: string[];
  rawRows: Record<string, string>[];
  reviewed: Set<string>;
  overrides: Record<string, QuestionOverride>;
  tags: Record<string, string[]>;
  details: RawDetailsBundle;
  csvMtimeMs: number;
  progressMtimeMs: number;
  detailsMtimeMs: number;
  overridesMtimeMs: number;
  tagsMtimeMs: number;
}

const globalForStore = globalThis as unknown as {
  __auditCache?: Map<string, CachedDataset>;
  __auditLocks?: Map<string, Promise<unknown>>;
};

const datasetCache =
  globalForStore.__auditCache ?? (globalForStore.__auditCache = new Map());
const datasetLocks =
  globalForStore.__auditLocks ?? (globalForStore.__auditLocks = new Map());

async function withLock<T>(dsId: string, fn: () => Promise<T>): Promise<T> {
  const prev = datasetLocks.get(dsId) ?? Promise.resolve();
  const next = prev.catch(() => undefined).then(fn);
  datasetLocks.set(dsId, next);
  return next;
}

async function fileExists(p: string): Promise<boolean> {
  try {
    const st = await fs.stat(p);
    return st.isFile();
  } catch {
    return false;
  }
}

async function fileMtimeMs(p: string): Promise<number> {
  try {
    const st = await fs.stat(p);
    return st.mtimeMs;
  } catch {
    return 0;
  }
}

export async function discoverDatasets(): Promise<{
  defaultDs: string;
  datasets: DatasetInfo[];
}> {
  const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
  const datasets: DatasetInfo[] = [];

  const addIfPresent = async (dirName: string, csvFile: string) => {
    const full = path.join(DATA_DIR, dirName, csvFile);
    if (await fileExists(full)) {
      const id = `${dirName}/${csvFile}`;
      if (!datasets.some((d) => d.id === id)) {
        datasets.push({
          id,
          name: dirName,
          csvFile,
          path: full,
        });
      }
    }
  };

  await addIfPresent("Twin-2K-500-Mega-Study", "question_audit.csv");

  const sortedDirs = entries
    .filter((e) => e.isDirectory() && e.name !== "audit-app" && !e.name.startsWith("."))
    .map((e) => e.name)
    .sort();

  for (const dirName of sortedDirs) {
    await addIfPresent(dirName, "question_audit.csv");
    await addIfPresent(dirName, "question_bank.csv");

    const subEntries = await fs.readdir(path.join(DATA_DIR, dirName), {
      withFileTypes: true,
    });
    const sortedSubDirs = subEntries
      .filter((e) => e.isDirectory() && !e.name.startsWith("."))
      .map((e) => e.name)
      .sort();
    for (const subName of sortedSubDirs) {
      const relDir = `${dirName}/${subName}`;
      await addIfPresent(relDir, "question_audit.csv");
      await addIfPresent(relDir, "question_bank.csv");
    }
  }

  if (datasets.length === 0) {
    throw new Error(`No question_audit.csv or question_bank.csv found under ${DATA_DIR}`);
  }

  return {
    defaultDs: datasets[0].id,
    datasets,
  };
}

function detectSchemaKind(columns: string[]): SchemaKind {
  const set = new Set(columns);
  if (
    set.has("catalog_idx") &&
    set.has("QuestionID") &&
    set.has("include") &&
    set.has("kev_type")
  ) {
    return "question_audit";
  }
  if (set.has("v") && set.has("type") && set.has("question") && set.has("options")) {
    return "question_bank";
  }
  throw new Error(`Unrecognized CSV column schema: ${columns.join(", ")}`);
}

async function resolveDatasetCsvPath(dsId: string): Promise<string> {
  const { datasets } = await discoverDatasets();
  const match = datasets.find((d) => d.id === dsId);
  if (!match) {
    throw new Error(`Unknown dataset id: ${dsId}`);
  }
  return match.path;
}

const DETAILS_SCHEMA_VERSION = 2;

async function ensureDetailsBuilt(datasetDir: string, detailsPath: string): Promise<void> {
  if (await fileExists(detailsPath)) {
    try {
      const head = await fs.readFile(detailsPath, "utf8");
      const parsed = JSON.parse(head) as { version?: number };
      if (parsed.version === DETAILS_SCHEMA_VERSION) {
        return;
      }
    } catch {
      // Rebuild corrupted or outdated index below.
    }
  }
  await execFileAsync(PYTHON_BIN, [BUILD_INDEX_SCRIPT, "--dataset-dir", datasetDir], {
    maxBuffer: 16 * 1024 * 1024,
  });
}

async function loadOrRefreshDataset(dsId: string): Promise<CachedDataset> {
  const csvPath = await resolveDatasetCsvPath(dsId);
  const datasetDir = path.dirname(csvPath);
  const stem = path.basename(csvPath, ".csv");
  const progressPath = path.join(datasetDir, `.${stem}.review_progress.json`);
  const detailsPath = path.join(datasetDir, DETAILS_FILE);
  const overridesPath = path.join(datasetDir, OVERRIDES_FILE);
  const tagsPath = path.join(datasetDir, TAGS_FILE);

  const curCsvMtime = await fileMtimeMs(csvPath);
  const curProgMtime = await fileMtimeMs(progressPath);
  const curDetailsMtime = await fileMtimeMs(detailsPath);
  const curOverridesMtime = await fileMtimeMs(overridesPath);
  const curTagsMtime = await fileMtimeMs(tagsPath);
  const existing = datasetCache.get(dsId);

  if (
    existing &&
    existing.details.version === DETAILS_SCHEMA_VERSION &&
    existing.overrides !== undefined &&
    existing.overridesPath !== undefined &&
    existing.tags !== undefined &&
    existing.tagsPath !== undefined &&
    existing.csvMtimeMs === curCsvMtime &&
    existing.progressMtimeMs === curProgMtime &&
    existing.detailsMtimeMs === curDetailsMtime &&
    existing.overridesMtimeMs === curOverridesMtime &&
    existing.tagsMtimeMs === curTagsMtime
  ) {
    return existing;
  }

  await ensureDetailsBuilt(datasetDir, detailsPath);
  const finalDetailsMtime = await fileMtimeMs(detailsPath);

  const [csvText, detailsText, hasNotebook, hasOverrides, hasTags] =
    await Promise.all([
      fs.readFile(csvPath, "utf8"),
      fs.readFile(detailsPath, "utf8"),
      fileExists(path.join(datasetDir, "data.ipynb")),
      fileExists(overridesPath),
      fileExists(tagsPath),
    ]);

  const { columns, rows: rawRows } = parseCsv(csvText);
  const schemaKind = detectSchemaKind(columns);
  const details = JSON.parse(detailsText) as RawDetailsBundle;

  let reviewed = new Set<string>();
  if (await fileExists(progressPath)) {
    const progJson = JSON.parse(await fs.readFile(progressPath, "utf8")) as {
      reviewed: string[];
    };
    reviewed = new Set(progJson.reviewed);
  }

  let overrides: Record<string, QuestionOverride> = {};
  if (hasOverrides) {
    const parsed = JSON.parse(await fs.readFile(overridesPath, "utf8")) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`${overridesPath} must contain a top-level JSON object`);
    }
    overrides = parsed as Record<string, QuestionOverride>;
  }

  const tags: Record<string, string[]> = {};
  if (hasTags) {
    const parsed = JSON.parse(await fs.readFile(tagsPath, "utf8")) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error(`${tagsPath} must contain a top-level JSON object`);
    }
    for (const [rk, rawList] of Object.entries(
      parsed as Record<string, unknown>
    )) {
      if (Array.isArray(rawList)) {
        const clean = normalizeTagList(rawList as string[]);
        if (clean.length > 0) {
          tags[rk] = clean;
        }
      }
    }
  }

  const cached: CachedDataset = {
    dsId,
    csvPath,
    datasetDir,
    progressPath,
    detailsPath,
    overridesPath,
    tagsPath,
    hasNotebook,
    schemaKind,
    columns,
    rawRows,
    reviewed,
    overrides,
    tags,
    details,
    csvMtimeMs: curCsvMtime,
    progressMtimeMs: curProgMtime,
    detailsMtimeMs: finalDetailsMtime,
    overridesMtimeMs: curOverridesMtime,
    tagsMtimeMs: curTagsMtime,
  };
  datasetCache.set(dsId, cached);
  return cached;
}

function buildRowObject(ds: CachedDataset, rowIdx: number): AuditRow {
  const data = ds.rawRows[rowIdx];
  const rowKey = makeRowKey(ds.schemaKind, ds.columns, data);
  const detail = ds.details.items[rowKey];
  const override = ds.overrides[rowKey];
  const userTags = ds.tags[rowKey];
  return classifyAuditRow(
    rowIdx,
    ds.schemaKind,
    ds.columns,
    data,
    ds.reviewed.has(rowKey),
    detail,
    override,
    userTags
  );
}

export async function getDatasetPayload(dsId: string): Promise<DatasetPayload> {
  return withLock(dsId, async () => {
    const ds = await loadOrRefreshDataset(dsId);
    const rows: AuditRow[] = ds.rawRows.map((_, i) => buildRowObject(ds, i));
    return {
      dsId: ds.dsId,
      csvPath: ds.csvPath,
      overridesPath: ds.overridesPath,
      tagsPath: ds.tagsPath,
      schemaKind: ds.schemaKind,
      columns: ds.columns,
      hasNotebook: ds.hasNotebook,
      rows,
      items: ds.details.items,
      studyContexts: ds.details.study_contexts,
      overrides: ds.overrides,
      tags: ds.tags,
    };
  });
}

function applyPatchToRow(
  ds: CachedDataset,
  rowIdx: number,
  patch: Record<string, string>
): void {
  if (rowIdx < 0 || rowIdx >= ds.rawRows.length) {
    throw new Error(`rowIdx ${rowIdx} out of bounds [0, ${ds.rawRows.length})`);
  }
  const cur = { ...ds.rawRows[rowIdx] };
  for (const [k, v] of Object.entries(patch)) {
    if (!ds.columns.includes(k)) {
      throw new Error(`Column "${k}" does not exist in ${path.basename(ds.csvPath)}`);
    }
    cur[k] = String(v);
  }

  if (ds.schemaKind === "question_audit") {
    if (cur.include !== "True" && cur.include !== "False") {
      throw new Error(`include must be "True" or "False", got "${cur.include}"`);
    }
    if (cur.include === "True") {
      if (cur.kev_type !== "score" && cur.kev_type !== "choice") {
        throw new Error(
          `Included question must have kev_type "score" or "choice", got "${cur.kev_type}"`
        );
      }
    } else {
      cur.kev_type = "";
    }

    if (ds.columns.includes("context_source")) {
      const cs = cur.context_source.trim();
      if (
        cs !== "" &&
        cs !== "in_block_db" &&
        !cs.startsWith("qid:") &&
        !cs.startsWith("prefix:")
      ) {
        throw new Error(
          `context_source must be "", "in_block_db", "qid:<QID>", or "prefix:<text>", got "${cs}"`
        );
      }
      cur.context_source = cs;
      if (
        ds.columns.includes("Context_preview") &&
        "context_source" in patch
      ) {
        const key = makeRowKey(ds.schemaKind, ds.columns, cur);
        const ctxKey = cur.study || "__all__";
        cur.Context_preview = resolveContextText(
          cs,
          ds.details.items[key],
          ds.details.study_contexts[ctxKey]
        );
      }
    }
  } else {
    if (!["score", "choice", "noul", "exclude"].includes(cur.type)) {
      throw new Error(`type must be score/choice/noul/exclude, got "${cur.type}"`);
    }
    if (
      ds.columns.includes("reversed") &&
      !["True", "False", ""].includes(cur.reversed)
    ) {
      throw new Error(`reversed must be "True", "False", or "", got "${cur.reversed}"`);
    }
  }

  ds.rawRows[rowIdx] = cur;
}

async function saveCsvAtomic(ds: CachedDataset): Promise<void> {
  const tmpPath = `${ds.csvPath}.tmp`;
  const serialized = stringifyCsv(ds.columns, ds.rawRows);
  await fs.writeFile(tmpPath, serialized, "utf8");
  await fs.rename(tmpPath, ds.csvPath);
  ds.csvMtimeMs = await fileMtimeMs(ds.csvPath);
}

async function saveProgressAtomic(ds: CachedDataset): Promise<void> {
  const tmpPath = `${ds.progressPath}.tmp`;
  const payload = JSON.stringify(
    { reviewed: Array.from(ds.reviewed).sort() },
    null,
    2
  ) + "\n";
  await fs.writeFile(tmpPath, payload, "utf8");
  await fs.rename(tmpPath, ds.progressPath);
  ds.progressMtimeMs = await fileMtimeMs(ds.progressPath);
}

function applyOverrideToRow(
  ds: CachedDataset,
  rowIdx: number,
  rawOverride: QuestionOverride | null
): void {
  if (rowIdx < 0 || rowIdx >= ds.rawRows.length) {
    throw new Error(`rowIdx ${rowIdx} out of bounds [0, ${ds.rawRows.length})`);
  }
  const rowData = ds.rawRows[rowIdx];
  const rowKey = makeRowKey(ds.schemaKind, ds.columns, rowData);
  const detail = ds.details.items[rowKey];
  const cleaned = normalizeQuestionOverride(
    rawOverride,
    ds.schemaKind,
    rowData,
    detail
  );

  if (!cleaned) {
    delete ds.overrides[rowKey];
    return;
  }

  if (
    cleaned.stem !== undefined &&
    ds.schemaKind === "question_audit" &&
    rowData.catalog_idx?.includes("..")
  ) {
    throw new Error(
      `Cannot apply a single stem override to collapsed multi-item loop ${rowKey}`
    );
  }

  if (cleaned.subitems) {
    const nRows = detail ? detail.rows.length : 0;
    const isMultiBipolar = Boolean(
      detail &&
        detail.QuestionType === "Matrix" &&
        detail.Selector === "Bipolar" &&
        detail.rows.length > 1
    );
    for (const [k, val] of Object.entries(cleaned.subitems)) {
      const idx = Number(k);
      if (!Number.isInteger(idx) || idx < 0 || idx >= nRows) {
        throw new Error(
          `Sub-item index ${k} out of range [0, ${nRows}) for ${rowKey}`
        );
      }
      if (!val.trim()) {
        throw new Error(
          `Sub-item #${idx + 1} override on ${rowKey} cannot be empty`
        );
      }
      if (isMultiBipolar && !val.includes(":")) {
        throw new Error(
          `Bipolar sub-item #${idx + 1} override on ${rowKey} must preserve "Left:Right" format`
        );
      }
    }
  }

  if (cleaned.options) {
    const knownOpts =
      detail && detail.options.length > 0
        ? detail.options.map((o) => o.option)
        : (rowData.Options_preview || rowData.options || "")
            .replace(/^\[\d+\]\s*/, "")
            .split(" | ")
            .map((s) => s.trim())
            .filter(Boolean);
    const knownSet = new Set(knownOpts);
    for (const [orig, repl] of Object.entries(cleaned.options)) {
      if (!knownSet.has(orig)) {
        throw new Error(`Unknown raw option "${orig}" on ${rowKey}`);
      }
      if (!repl.trim()) {
        throw new Error(
          `Replacement option for "${orig}" on ${rowKey} cannot be empty`
        );
      }
    }
    const dropSet = new Set(
      (rowData.drop_options || "")
        .split("|")
        .map((s) => s.trim())
        .filter(Boolean)
    );
    const optionGroups =
      detail && detail.opt_variant_sets && detail.opt_variant_sets.length > 1
        ? detail.opt_variant_sets
        : [knownOpts];
    for (const grp of optionGroups) {
      const finalActive = grp
        .filter((o) => !dropSet.has(o))
        .map((o) => cleaned.options?.[o] ?? o);
      if (new Set(finalActive).size !== finalActive.length) {
        throw new Error(
          `Option overrides on ${rowKey} produce duplicate option labels`
        );
      }
    }
  }

  ds.overrides[rowKey] = cleaned;
}

async function saveOverridesAtomic(ds: CachedDataset): Promise<void> {
  const sortedKeys = Object.keys(ds.overrides).sort((a, b) => {
    const aNum = parseInt(a.split(":", 1)[0], 10);
    const bNum = parseInt(b.split(":", 1)[0], 10);
    if (!Number.isNaN(aNum) && !Number.isNaN(bNum) && aNum !== bNum) {
      return aNum - bNum;
    }
    return a.localeCompare(b);
  });

  if (sortedKeys.length === 0) {
    if (await fileExists(ds.overridesPath)) {
      await fs.unlink(ds.overridesPath);
    }
    ds.overridesMtimeMs = 0;
    return;
  }

  const ordered: Record<string, QuestionOverride> = {};
  for (const k of sortedKeys) {
    ordered[k] = ds.overrides[k];
  }
  const tmpPath = `${ds.overridesPath}.tmp`;
  const payload = JSON.stringify(ordered, null, 2) + "\n";
  await fs.writeFile(tmpPath, payload, "utf8");
  await fs.rename(tmpPath, ds.overridesPath);
  ds.overridesMtimeMs = await fileMtimeMs(ds.overridesPath);
}

function applyTagsToRow(
  ds: CachedDataset,
  rowIdx: number,
  rawTags: string[] | null
): void {
  if (rowIdx < 0 || rowIdx >= ds.rawRows.length) {
    throw new Error(`rowIdx ${rowIdx} out of bounds [0, ${ds.rawRows.length})`);
  }
  const rowData = ds.rawRows[rowIdx];
  const rowKey = makeRowKey(ds.schemaKind, ds.columns, rowData);
  const cleaned = normalizeTagList(rawTags);
  if (cleaned.length === 0) {
    delete ds.tags[rowKey];
  } else {
    ds.tags[rowKey] = cleaned;
  }
}

async function saveTagsAtomic(ds: CachedDataset): Promise<void> {
  const sortedKeys = Object.keys(ds.tags).sort((a, b) => {
    const aNum = parseInt(a.split(":", 1)[0], 10);
    const bNum = parseInt(b.split(":", 1)[0], 10);
    if (!Number.isNaN(aNum) && !Number.isNaN(bNum) && aNum !== bNum) {
      return aNum - bNum;
    }
    return a.localeCompare(b);
  });

  if (sortedKeys.length === 0) {
    if (await fileExists(ds.tagsPath)) {
      await fs.unlink(ds.tagsPath);
    }
    ds.tagsMtimeMs = 0;
    return;
  }

  const ordered: Record<string, string[]> = {};
  for (const k of sortedKeys) {
    ordered[k] = ds.tags[k];
  }
  const tmpPath = `${ds.tagsPath}.tmp`;
  const payload = JSON.stringify(ordered, null, 2) + "\n";
  await fs.writeFile(tmpPath, payload, "utf8");
  await fs.rename(tmpPath, ds.tagsPath);
  ds.tagsMtimeMs = await fileMtimeMs(ds.tagsPath);
}

export interface RowUpdateEntry {
  rowIdx: number;
  patch: Record<string, string>;
  override?: QuestionOverride | null;
  tags?: string[] | null;
  reviewed?: boolean;
}

export async function updateRowEntries(
  dsId: string,
  entries: RowUpdateEntry[]
): Promise<{
  updated: AuditRow[];
  overrides: Record<string, QuestionOverride>;
  tags: Record<string, string[]>;
}> {
  return withLock(dsId, async () => {
    const ds = await loadOrRefreshDataset(dsId);
    let touchedCsv = false;
    let touchedOverrides = false;
    let touchedTags = false;
    let touchedProgress = false;

    for (const entry of entries) {
      if (Object.keys(entry.patch).length > 0) {
        applyPatchToRow(ds, entry.rowIdx, entry.patch);
        touchedCsv = true;
      }
      if (entry.override !== undefined) {
        applyOverrideToRow(ds, entry.rowIdx, entry.override);
        touchedOverrides = true;
      }
      if (entry.tags !== undefined) {
        applyTagsToRow(ds, entry.rowIdx, entry.tags);
        touchedTags = true;
      }
      if (entry.reviewed !== undefined) {
        const key = makeRowKey(
          ds.schemaKind,
          ds.columns,
          ds.rawRows[entry.rowIdx]
        );
        if (entry.reviewed) {
          ds.reviewed.add(key);
        } else {
          ds.reviewed.delete(key);
        }
        touchedProgress = true;
      }
    }

    if (touchedCsv) {
      await saveCsvAtomic(ds);
    }
    if (touchedOverrides) {
      await saveOverridesAtomic(ds);
    }
    if (touchedTags) {
      await saveTagsAtomic(ds);
    }
    if (touchedProgress) {
      await saveProgressAtomic(ds);
    }

    return {
      updated: entries.map((e) => buildRowObject(ds, e.rowIdx)),
      overrides: ds.overrides,
      tags: ds.tags,
    };
  });
}

export async function updateRows(
  dsId: string,
  rowIndices: number[],
  patch: Record<string, string>,
  reviewed?: boolean
): Promise<{
  updated: AuditRow[];
  overrides: Record<string, QuestionOverride>;
  tags: Record<string, string[]>;
}> {
  return updateRowEntries(
    dsId,
    rowIndices.map((rowIdx) => ({ rowIdx, patch, reviewed }))
  );
}

export async function rebuildDatasetParquet(dsId: string): Promise<{
  ok: boolean;
  summary: string;
  stderr: string;
}> {
  return withLock(dsId, async () => {
    const ds = await loadOrRefreshDataset(dsId);
    const nbPath = path.join(ds.datasetDir, "data.ipynb");
    if (!(await fileExists(nbPath))) {
      throw new Error(`No data.ipynb found in ${ds.datasetDir}`);
    }

    try {
      const venvBinDir = path.dirname(PYTHON_BIN);
      const execEnv = {
        ...process.env,
        PATH: `${venvBinDir}${path.delimiter}${process.env.PATH ?? ""}`,
      };
      await execFileAsync(
        PYTHON_BIN,
        [
          "-c",
          `import nbformat
from nbclient import NotebookClient
nb = nbformat.read("data.ipynb", as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
nbformat.write(nb, "data.ipynb")`,
        ],
        {
          cwd: ds.datasetDir,
          env: execEnv,
          maxBuffer: 32 * 1024 * 1024,
        }
      );
      // Re-index details in case *_US_long.parquet counts changed
      await execFileAsync(
        PYTHON_BIN,
        [BUILD_INDEX_SCRIPT, "--dataset-dir", ds.datasetDir],
        { env: execEnv, maxBuffer: 16 * 1024 * 1024 }
      );
      datasetCache.delete(dsId);

      const { stdout } = await execFileAsync(
        PYTHON_BIN,
        [
          "-c",
          `import glob, os, pandas as pd
pqs = sorted(glob.glob("*_US_long.parquet"))
if pqs:
    df = pd.read_parquet(pqs[0], columns=["CASEID", "v", "type"])
    nc = df[df["type"] == "choice"]["v"].nunique()
    ns = df[df["type"] == "score"]["v"].nunique()
    print(f"{pqs[0]}: {len(df):,} rows | {df['CASEID'].nunique():,} respondents | {df['v'].nunique():,} questions ({nc:,} choice, {ns:,} score)")
else:
    print("Notebook executed.")`,
        ],
        { cwd: ds.datasetDir }
      );

      return { ok: true, summary: stdout.trim(), stderr: "" };
    } catch (err: unknown) {
      const e = err as { stderr?: string; message?: string };
      return {
        ok: false,
        summary: "",
        stderr: (e.stderr || e.message || String(err)).slice(-2500),
      };
    }
  });
}
