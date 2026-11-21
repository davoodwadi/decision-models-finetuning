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
  PersonaCompletenessFilter,
  PersonasPagePayload,
  PersonaSummaryBundle,
  QuestionOverride,
  RespondentQuestionsPayload,
  SchemaKind,
} from "./types";

const execFileAsync = promisify(execFile);

const APP_DIR = process.cwd();
const DATA_DIR = path.resolve(APP_DIR, "..");
const REPO_ROOT = path.resolve(DATA_DIR, "..", "..");
const PYTHON_BIN = path.join(REPO_ROOT, ".venv-cu13", "bin", "python");
const BUILD_INDEX_SCRIPT = path.join(APP_DIR, "scripts", "build_index.py");
const INSPECT_PERSONAS_SCRIPT = path.join(APP_DIR, "scripts", "inspect_personas.py");
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

interface CachedPersonas {
  dsId: string;
  parquetPath: string;
  parquetMtimeMs: number;
  bundle: PersonaSummaryBundle;
}

const globalForStore = globalThis as unknown as {
  __auditCache?: Map<string, CachedDataset>;
  __auditLocks?: Map<string, Promise<unknown>>;
  __personaCache?: Map<string, CachedPersonas>;
};

const datasetCache =
  globalForStore.__auditCache ?? (globalForStore.__auditCache = new Map());
const datasetLocks =
  globalForStore.__auditLocks ?? (globalForStore.__auditLocks = new Map());
const personaCache =
  globalForStore.__personaCache ?? (globalForStore.__personaCache = new Map());

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

  const [csvText, detailsText, hasDataNb, hasStimulus, hasOverrides, hasTags] =
    await Promise.all([
      fs.readFile(csvPath, "utf8"),
      fs.readFile(detailsPath, "utf8"),
      fileExists(path.join(datasetDir, "data.ipynb")),
      fileExists(path.join(datasetDir, "stimulus.py")),
      fileExists(overridesPath),
      fileExists(tagsPath),
    ]);
  const hasNotebook =
    hasDataNb || (hasStimulus && PYTHON_EXPORT_DIRS.includes(path.basename(datasetDir)));

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
    detail,
    ds.dsId
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

  if (cleaned.subitem_sep !== undefined) {
    if (ds.dsId.startsWith("Twin-2K-500-Mega-Study/")) {
      throw new Error(
        `subitem_sep override is disabled on verified dataset ${ds.dsId}`
      );
    }
    const isMatrixLikert = Boolean(
      detail &&
        detail.QuestionType === "Matrix" &&
        detail.Selector === "Likert" &&
        detail.rows.length > 0
    );
    if (!isMatrixLikert) {
      throw new Error(
        `subitem_sep override on ${rowKey} is only valid on Matrix Likert questions`
      );
    }
    if (!cleaned.subitem_sep.length) {
      throw new Error(`subitem_sep override on ${rowKey} cannot be empty`);
    }
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

// Dataset folders whose parquet is written by `data.<folder>.export_<folder>_parquet()` (others use data.ipynb).
const PYTHON_EXPORT_DIRS = ["grummon2023", "grummon2025", "covid", "Kriner2026", "berke2023", "wu2025", "kantorowicz2026", "magistro2025"];

// Folders that must be rebuilt before the others: the Twin-2K-500 Mega-Study notebook checks its personas against
// Twin-2K-500's parquet, so Twin-2K-500 goes first.
const REBUILD_FIRST_DIRS = ["Twin-2K-500"];

export interface RebuildResult {
  dsId: string;
  ok: boolean;
  summary: string;
  stderr: string;
}

/** Rebuild every dataset that has an exporter (data.ipynb or Python), in dependency order; stop at the first failure. */
export async function rebuildAllParquets(): Promise<{ ok: boolean; results: RebuildResult[]; skipped: string[] }> {
  const { datasets } = await discoverDatasets();
  const rebuildable: { dsId: string; dirName: string }[] = [];
  for (const ds of datasets) {
    const dir = path.dirname(ds.path);
    const dirName = path.basename(dir);
    const hasNb = await fileExists(path.join(dir, "data.ipynb"));
    const hasStimulus = await fileExists(path.join(dir, "stimulus.py"));
    if (hasNb || (hasStimulus && PYTHON_EXPORT_DIRS.includes(dirName))) {
      rebuildable.push({ dsId: ds.id, dirName });
    }
  }
  const rank = (d: string) => (REBUILD_FIRST_DIRS.includes(d) ? REBUILD_FIRST_DIRS.indexOf(d) : REBUILD_FIRST_DIRS.length);
  rebuildable.sort((a, b) => rank(a.dirName) - rank(b.dirName) || a.dsId.localeCompare(b.dsId));

  const results: RebuildResult[] = [];
  for (let i = 0; i < rebuildable.length; i++) {
    const res = await rebuildDatasetParquet(rebuildable[i].dsId);
    results.push({ dsId: rebuildable[i].dsId, ...res });
    if (!res.ok) {
      return { ok: false, results, skipped: rebuildable.slice(i + 1).map((r) => r.dsId) };
    }
  }
  return { ok: true, results, skipped: [] };
}

export async function rebuildDatasetParquet(dsId: string): Promise<{
  ok: boolean;
  summary: string;
  stderr: string;
}> {
  return withLock(dsId, async () => {
    const ds = await loadOrRefreshDataset(dsId);
    const nbPath = path.join(ds.datasetDir, "data.ipynb");
    const hasNb = await fileExists(nbPath);
    const dirBase = path.basename(ds.datasetDir);
    const isPythonExport = PYTHON_EXPORT_DIRS.includes(dirBase);
    if (!hasNb && !isPythonExport) {
      throw new Error(
        `No data.ipynb or Python parquet exporter found in ${ds.datasetDir}`
      );
    }

    try {
      const venvBinDir = path.dirname(PYTHON_BIN);
      const codeDir = path.resolve(DATA_DIR, "..");
      const execEnv = {
        ...process.env,
        PATH: `${venvBinDir}${path.delimiter}${process.env.PATH ?? ""}`,
        PYTHONPATH: codeDir,
      };
      if (isPythonExport) {
        await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.${dirBase} import export_${dirBase}_parquet; export_${dirBase}_parquet()`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
      } else {
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
      }
      // Re-index details in case *_US_long.parquet counts changed
      await execFileAsync(
        PYTHON_BIN,
        [BUILD_INDEX_SCRIPT, "--dataset-dir", ds.datasetDir],
        { env: execEnv, maxBuffer: 16 * 1024 * 1024 }
      );
      datasetCache.delete(dsId);
      personaCache.delete(dsId);

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

async function findLongParquetPath(datasetDir: string): Promise<string> {
  const entries = await fs.readdir(datasetDir, { withFileTypes: true });
  const matches = entries
    .filter((e) => e.isFile() && e.name.endsWith("_long.parquet"))
    .map((e) => e.name)
    .sort();
  if (matches.length === 0) {
    throw new Error(
      `No *_long.parquet found in ${datasetDir}. Click "Parquet -> This dataset" to build it first.`
    );
  }
  if (matches.length > 1) {
    throw new Error(
      `Expected a single *_long.parquet in ${datasetDir}, found: ${matches.join(", ")}`
    );
  }
  return path.join(datasetDir, matches[0]);
}

async function loadOrRefreshPersonas(dsId: string): Promise<PersonaSummaryBundle> {
  const csvPath = await resolveDatasetCsvPath(dsId);
  const datasetDir = path.dirname(csvPath);
  const parquetPath = await findLongParquetPath(datasetDir);
  const curMtime = await fileMtimeMs(parquetPath);
  const cached = personaCache.get(dsId);

  if (
    cached &&
    cached.parquetPath === parquetPath &&
    cached.parquetMtimeMs === curMtime
  ) {
    return cached.bundle;
  }

  const { stdout } = await execFileAsync(
    PYTHON_BIN,
    [INSPECT_PERSONAS_SCRIPT, "--dataset-dir", datasetDir, "--mode", "summary"],
    { maxBuffer: 64 * 1024 * 1024 }
  );
  const bundle = JSON.parse(stdout) as PersonaSummaryBundle;
  personaCache.set(dsId, {
    dsId,
    parquetPath,
    parquetMtimeMs: curMtime,
    bundle,
  });
  return bundle;
}

export interface PersonaQueryOptions {
  search?: string;
  attrCol?: string;
  attrVal?: string;
  completeness?: PersonaCompletenessFilter;
  page?: number;
  pageSize?: number;
}

export async function getPersonasPage(
  dsId: string,
  opts: PersonaQueryOptions = {}
): Promise<PersonasPagePayload> {
  const bundle = await loadOrRefreshPersonas(dsId);
  const searchLower = (opts.search ?? "").trim().toLowerCase();
  const attrCol = (opts.attrCol ?? "").trim();
  const attrVal = opts.attrVal ?? "";
  const completeness: PersonaCompletenessFilter =
    opts.completeness === "complete" || opts.completeness === "incomplete"
      ? opts.completeness
      : "all";
  const pageSize = Math.min(200, Math.max(10, opts.pageSize ?? 100));

  let completeCount = 0;
  let incompleteCount = 0;
  for (const r of bundle.respondents) {
    if (r.nPresent >= bundle.nAttributes) {
      completeCount += 1;
    } else {
      incompleteCount += 1;
    }
  }

  const filtered = bundle.respondents.filter((r) => {
    if (completeness === "complete" && r.nPresent < bundle.nAttributes) {
      return false;
    }
    if (completeness === "incomplete" && r.nPresent >= bundle.nAttributes) {
      return false;
    }
    if (attrCol && attrVal !== "") {
      const val = r.values[attrCol] ?? null;
      if (attrVal === "__MISSING__") {
        if (val !== null) return false;
      } else if (val !== attrVal) {
        return false;
      }
    }
    if (searchLower) {
      if (r.caseId.toLowerCase().includes(searchLower)) return true;
      for (const v of Object.values(r.extraIds)) {
        if (v && v.toLowerCase().includes(searchLower)) return true;
      }
      for (const v of Object.values(r.values)) {
        if (v && v.toLowerCase().includes(searchLower)) return true;
      }
      return false;
    }
    return true;
  });

  const totalFiltered = filtered.length;
  const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize));
  const page = Math.min(totalPages, Math.max(1, opts.page ?? 1));
  const start = (page - 1) * pageSize;
  const respondents = filtered.slice(start, start + pageSize);

  return {
    dsId,
    parquetFile: bundle.parquetFile,
    parquetMtimeMs: bundle.parquetMtimeMs,
    loaderKind: bundle.loaderKind,
    isspRuntimeMapped: bundle.isspRuntimeMapped,
    nRespondents: bundle.nRespondents,
    nRows: bundle.nRows,
    nQuestions: bundle.nQuestions,
    extraIdCols: bundle.extraIdCols,
    hasCountry: bundle.hasCountry,
    nAttributes: bundle.nAttributes,
    uniquePersonasWithCountry: bundle.uniquePersonasWithCountry,
    uniquePersonasWithoutCountry: bundle.uniquePersonasWithoutCountry,
    avgAttrsPresent: bundle.avgAttrsPresent,
    minAttrsPresent: bundle.minAttrsPresent,
    maxAttrsPresent: bundle.maxAttrsPresent,
    completeCount,
    incompleteCount,
    attributes: bundle.attributes,
    totalFiltered,
    page,
    pageSize,
    totalPages,
    respondents,
  };
}

export async function getRespondentQuestions(
  dsId: string,
  caseId: string,
  limit = 50
): Promise<RespondentQuestionsPayload> {
  const csvPath = await resolveDatasetCsvPath(dsId);
  const datasetDir = path.dirname(csvPath);
  const safeLimit = Math.min(200, Math.max(1, Math.floor(limit)));
  const { stdout } = await execFileAsync(
    PYTHON_BIN,
    [
      INSPECT_PERSONAS_SCRIPT,
      "--dataset-dir",
      datasetDir,
      "--mode",
      "questions",
      "--caseid",
      caseId,
      "--limit",
      String(safeLimit),
    ],
    { maxBuffer: 32 * 1024 * 1024 }
  );
  const parsed = JSON.parse(stdout) as Omit<RespondentQuestionsPayload, "dsId">;
  return { dsId, ...parsed };
}
