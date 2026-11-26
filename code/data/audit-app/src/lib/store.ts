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
  MarginalAnswerOption,
  MarginalGroupRow,
  MarginalPersonaOption,
  MarginalSeparationStats,
  PersonaCompletenessFilter,
  PersonasPagePayload,
  PersonaSummaryBundle,
  QuestionMarginalsPayload,
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

interface RawMarginalsQuestion {
  questionType: string;
  reversed: boolean;
  optionCodes: number[];
  optionTexts: string[];
  arms: string[];
  byArm: Record<string, Record<string, number[]>>;
}

interface RawMarginalsBundle {
  parquetFile: string;
  loaderKind: string;
  isMarginal?: boolean;
  availablePersonas: MarginalPersonaOption[];
  selectedPersona: MarginalPersonaOption | null;
  canonicalLevelOrder: string[] | null;
  allVs: string[];
  questions: Record<string, RawMarginalsQuestion>;
}

interface CachedMarginals {
  parquetPath: string;
  parquetMtimeMs: number;
  bundle: RawMarginalsBundle;
}

const globalForStore = globalThis as unknown as {
  __auditCache?: Map<string, CachedDataset>;
  __auditLocks?: Map<string, Promise<unknown>>;
  __personaCache?: Map<string, CachedPersonas>;
  __marginalsCache?: Map<string, CachedMarginals>;
};

const datasetCache =
  globalForStore.__auditCache ?? (globalForStore.__auditCache = new Map());
const datasetLocks =
  globalForStore.__auditLocks ?? (globalForStore.__auditLocks = new Map());
const personaCache =
  globalForStore.__personaCache ?? (globalForStore.__personaCache = new Map());
const marginalsCache =
  globalForStore.__marginalsCache ?? (globalForStore.__marginalsCache = new Map());

function clearMarginalsCacheForDataset(dsId: string): void {
  const prefix = `${dsId}::`;
  for (const key of Array.from(marginalsCache.keys())) {
    if (key.startsWith(prefix)) {
      marginalsCache.delete(key);
    }
  }
}

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

  const addIfPresent = async (dirName: string, csvFile: string): Promise<boolean> => {
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
      return true;
    }
    return false;
  };

  await addIfPresent("Twin-2K-500-Mega-Study", "question_audit.csv");

  const sortedDirs = entries
    .filter((e) => e.isDirectory() && e.name !== "audit-app" && !e.name.startsWith("."))
    .map((e) => e.name)
    .sort();

  for (const dirName of sortedDirs) {
    await addIfPresent(dirName, "question_audit.csv");

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
    }
  }

  if (datasets.length === 0) {
    throw new Error(`No question_audit.csv found under ${DATA_DIR}`);
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
  throw new Error(
    `Unsupported CSV column schema (expected question_audit.csv): ${columns.join(", ")}`
  );
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

/** Datasets whose .audit_details.json must carry per-item is_persona flags. */
function hasPersonaFlags(dirName: string): boolean {
  return dirName === WVS7_DIR || dirName === HINTS_DIR || dirName === SHED_DIR;
}

/** ISSP/WVS/HINTS details must carry the computed export checks (and persona flags for WVS/HINTS); older caches are rebuilt. */
function detailsHaveRequiredChecks(
  dirName: string,
  parsed: {
    items?: Record<
      string,
      {
        export_problem_if_included?: unknown;
        n_valid_by_country?: unknown;
        is_persona?: unknown;
        option_relabeled?: unknown;
      }
    >;
  }
): boolean {
  if (!parsed.items) {
    return false;
  }
  const needExport = hasExportChecks(dirName);
  const needPersona = hasPersonaFlags(dirName);
  return Object.values(parsed.items).every(
    (item) =>
      (!needExport ||
        (item.export_problem_if_included !== undefined &&
          item.n_valid_by_country !== undefined &&
          typeof item.option_relabeled === "boolean")) &&
      (!needPersona || typeof item.is_persona === "boolean")
  );
}

async function ensureDetailsBuilt(datasetDir: string, detailsPath: string): Promise<void> {
  const dirName = path.basename(datasetDir);
  const needsChecks = hasExportChecks(dirName) || hasPersonaFlags(dirName);
  if (await fileExists(detailsPath)) {
    try {
      const head = await fs.readFile(detailsPath, "utf8");
      const parsed = JSON.parse(head) as {
        version?: number;
        items?: Record<string, { export_problem_if_included?: unknown; n_valid_by_country?: unknown; is_persona?: unknown }>;
      };
      if (
        parsed.version === DETAILS_SCHEMA_VERSION &&
        (!needsChecks || detailsHaveRequiredChecks(dirName, parsed))
      ) {
        return;
      }
    } catch {
      // Rebuild corrupted or outdated index below.
    }
  }
  await runBuildIndex(datasetDir);
  if (needsChecks) {
    const rebuilt = JSON.parse(await fs.readFile(detailsPath, "utf8")) as {
      items?: Record<string, { export_problem_if_included?: unknown; n_valid_by_country?: unknown; is_persona?: unknown }>;
    };
    if (!detailsHaveRequiredChecks(dirName, rebuilt)) {
      throw new Error(`${detailsPath}: items are missing required export/persona checks after re-indexing`);
    }
  }
}

async function runBuildIndex(datasetDir: string): Promise<void> {
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
    detailsHaveRequiredChecks(path.basename(datasetDir), existing.details) &&
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
  const dirName = path.basename(datasetDir);
  const hasNotebook =
    hasDataNb ||
    ISSP_DIR_RE.test(dirName) ||
    dirName === WVS7_DIR ||
    dirName === HINTS_DIR ||
    dirName === SHED_DIR ||
    dirName in TWIN2K_EXPORT_FNS ||
    dirName in SUBPOP_EXPORT_SPLITS ||
    (hasStimulus && PYTHON_EXPORT_DIRS.includes(dirName));

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
    if (!["score", "choice", "exclude"].includes(cur.type)) {
      throw new Error(`type must be score/choice/exclude, got "${cur.type}"`);
    }
  }

  if (
    ds.columns.includes("reversed") &&
    !["True", "False", ""].includes(cur.reversed)
  ) {
    throw new Error(`reversed must be "True", "False", or "", got "${cur.reversed}"`);
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
const TWIN2K_EXPORT_FNS: Record<string, string> = {
  "Twin-2K-500": "export_twin2k_parquet",
  "Twin-2K-500-Mega-Study": "export_twin2k_megastudy_parquet",
};
const SUBPOP_EXPORT_SPLITS: Record<string, string> = {
  subpop_eval: "eval",
  subpop_train: "train",
};
const ISSP_DIR_RE = /^issp(\d{4})$/;
// WVS Wave 7 US: parquet written by data.wvs_export.export_wvs7(); details carry export checks like ISSP.
const WVS7_DIR = "WVS-Wave7";
// NCI HINTS 7 (2024): parquet written by data.hints7_export.export_hints7(); details carry export checks like WVS7.
const HINTS_DIR = "HINTS";
// Federal Reserve SHED 2025: parquet written by data.shed2025_export.export_shed2025(); details carry export checks like HINTS.
const SHED_DIR = "SHED";

/** Datasets whose .audit_details.json must carry export_problem_if_included / n_valid_by_country. */
function hasExportChecks(dirName: string): boolean {
  return ISSP_DIR_RE.test(dirName) || dirName === WVS7_DIR || dirName === HINTS_DIR || dirName === SHED_DIR;
}

export interface RebuildResult {
  dsId: string;
  ok: boolean;
  summary: string;
  stderr: string;
}

/** Rebuild every dataset that has an exporter (data.ipynb or Python); stop at the first failure. */
export async function rebuildAllParquets(): Promise<{ ok: boolean; results: RebuildResult[]; skipped: string[] }> {
  const { datasets } = await discoverDatasets();
  const rebuildable: { dsId: string; dirName: string }[] = [];
  for (const ds of datasets) {
    const dir = path.dirname(ds.path);
    const dirName = path.basename(dir);
    const hasNb = await fileExists(path.join(dir, "data.ipynb"));
    const hasStimulus = await fileExists(path.join(dir, "stimulus.py"));
    if (
      hasNb ||
      ISSP_DIR_RE.test(dirName) ||
      dirName === WVS7_DIR ||
      dirName === HINTS_DIR ||
      dirName === SHED_DIR ||
      dirName in TWIN2K_EXPORT_FNS ||
      dirName in SUBPOP_EXPORT_SPLITS ||
      (hasStimulus && PYTHON_EXPORT_DIRS.includes(dirName))
    ) {
      rebuildable.push({ dsId: ds.id, dirName });
    }
  }
  rebuildable.sort((a, b) => a.dsId.localeCompare(b.dsId));

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
    const isspMatch = ISSP_DIR_RE.exec(dirBase);
    const isWvs7 = dirBase === WVS7_DIR;
    const isHints7 = dirBase === HINTS_DIR;
    const isShed2025 = dirBase === SHED_DIR;
    const twin2kExportFn = TWIN2K_EXPORT_FNS[dirBase];
    const subpopSplit = SUBPOP_EXPORT_SPLITS[dirBase];
    const isPythonExport = PYTHON_EXPORT_DIRS.includes(dirBase);
    if (!hasNb && !isspMatch && !isWvs7 && !isHints7 && !isShed2025 && !twin2kExportFn && !subpopSplit && !isPythonExport) {
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
      let exportReport = "";
      if (isspMatch) {
        // Re-index first so the 'Blocks export' flags are current even when the export below fails.
        await runBuildIndex(ds.datasetDir);
        datasetCache.delete(dsId);
        const year = Number(isspMatch[1]);
        const exported = await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.issp_export import export_issp_wave, format_not_fielded
_, _, not_fielded = export_issp_wave(${year})
print(format_not_fielded(${year}, "US", not_fielded))`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
        exportReport = exported.stdout.trim();
      } else if (isWvs7) {
        // Re-index first so the 'Blocks export' flags are current even when the export below fails.
        await runBuildIndex(ds.datasetDir);
        datasetCache.delete(dsId);
        const exported = await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.wvs_export import export_wvs7, format_not_fielded
_, _, not_fielded = export_wvs7()
print(format_not_fielded(not_fielded))`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
        exportReport = exported.stdout.trim();
      } else if (isHints7) {
        await runBuildIndex(ds.datasetDir);
        datasetCache.delete(dsId);
        const exported = await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.hints7_export import export_hints7, format_not_fielded
_, _, not_fielded = export_hints7()
print(format_not_fielded(not_fielded))`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
        exportReport = exported.stdout.trim();
      } else if (isShed2025) {
        await runBuildIndex(ds.datasetDir);
        datasetCache.delete(dsId);
        const exported = await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.shed2025_export import export_shed2025, format_not_fielded
_, _, not_fielded = export_shed2025()
print(format_not_fielded(not_fielded))`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
        exportReport = exported.stdout.trim();
      } else if (twin2kExportFn) {
        await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.twin2k_export import ${twin2kExportFn}; ${twin2kExportFn}()`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
      } else if (subpopSplit) {
        await runBuildIndex(ds.datasetDir);
        datasetCache.delete(dsId);
        await execFileAsync(
          PYTHON_BIN,
          [
            "-c",
            `from data.subpop_export import export_subpop_parquet; export_subpop_parquet(${JSON.stringify(subpopSplit)})`,
          ],
          {
            cwd: ds.datasetDir,
            env: execEnv,
            maxBuffer: 32 * 1024 * 1024,
          }
        );
      } else if (isPythonExport) {
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
      await runBuildIndex(ds.datasetDir);
      datasetCache.delete(dsId);
      personaCache.delete(dsId);
      clearMarginalsCacheForDataset(dsId);

      const { stdout } = await execFileAsync(
        PYTHON_BIN,
        [
          "-c",
          `import glob, os, pandas as pd
pqs = sorted(glob.glob("*_US_long.parquet") + glob.glob("*_US_marginals.parquet"))
if pqs:
    df = pd.read_parquet(pqs[0], columns=["CASEID", "v", "type"])
    nc = df[df["type"] == "choice"]["v"].nunique()
    ns = df[df["type"] == "score"]["v"].nunique()
    unit = "subgroups" if pqs[0].endswith("_marginals.parquet") else "respondents"
    print(f"{pqs[0]}: {len(df):,} rows | {df['CASEID'].nunique():,} {unit} | {df['v'].nunique():,} questions ({nc:,} choice, {ns:,} score)")
else:
    print("Notebook executed.")`,
        ],
        { cwd: ds.datasetDir }
      );

      const summary = exportReport ? `${stdout.trim()} | ${exportReport}` : stdout.trim();
      return { ok: true, summary, stderr: "" };
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
    .filter(
      (e) =>
        e.isFile() &&
        (e.name.endsWith("_long.parquet") || e.name.endsWith("_marginals.parquet"))
    )
    .map((e) => e.name)
    .sort();
  if (matches.length === 0) {
    throw new Error(
      `No *_long.parquet or *_marginals.parquet found in ${datasetDir}. Click "Parquet -> This dataset" to build it first.`
    );
  }
  if (matches.length > 1) {
    throw new Error(
      `Expected a single dataset parquet in ${datasetDir}, found: ${matches.join(", ")}`
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
    isMarginal: bundle.isMarginal,
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

export interface QuestionMarginalsQuery {
  questionId: string;
  subitemId?: string;
  study?: string;
  persona?: string;
  arm?: string;
}

function matchQuestionVs(
  allVs: string[],
  questionId: string,
  subitemId?: string,
  study?: string
): string[] {
  const vSet = new Set(allVs);
  const slug = study ? study.replaceAll("_", "-") : undefined;

  if (subitemId) {
    const cands = slug ? [subitemId, `${slug}-${subitemId}`] : [subitemId];
    for (const c of cands) {
      if (vSet.has(c)) return [c];
    }
    if (slug) {
      const pref = `${slug}-${subitemId}_`;
      const hits = allVs.filter((v) => v.startsWith(pref));
      if (hits.length > 0) return hits;
    }
    const pref = `${subitemId}_`;
    const hits = allVs.filter((v) => v.startsWith(pref));
    if (hits.length > 0) return hits;
  }

  const cands = slug ? [questionId, `${slug}-${questionId}`] : [questionId];
  for (const c of cands) {
    if (vSet.has(c)) return [c];
  }
  if (slug) {
    const pref = `${slug}-${questionId}_`;
    const hits = allVs.filter((v) => v.startsWith(pref));
    if (hits.length > 0) return hits;
  }
  const pref = `${questionId}_`;
  return allVs.filter((v) => v.startsWith(pref));
}

async function loadOrRefreshMarginalsBundle(
  dsId: string,
  persona?: string
): Promise<RawMarginalsBundle> {
  const csvPath = await resolveDatasetCsvPath(dsId);
  const datasetDir = path.dirname(csvPath);
  const parquetPath = await findLongParquetPath(datasetDir);
  const curMtime = await fileMtimeMs(parquetPath);

  const reqKey = `${dsId}::${persona || "__default__"}`;
  const cached = marginalsCache.get(reqKey);
  if (
    cached &&
    cached.parquetPath === parquetPath &&
    cached.parquetMtimeMs === curMtime
  ) {
    return cached.bundle;
  }

  const args = [
    INSPECT_PERSONAS_SCRIPT,
    "--dataset-dir",
    datasetDir,
    "--mode",
    "marginals-bundle",
  ];
  if (persona) {
    args.push("--persona", persona);
  }

  const { stdout } = await execFileAsync(PYTHON_BIN, args, {
    maxBuffer: 64 * 1024 * 1024,
  });
  const bundle = JSON.parse(stdout) as RawMarginalsBundle;
  const entry: CachedMarginals = {
    parquetPath,
    parquetMtimeMs: curMtime,
    bundle,
  };
  marginalsCache.set(reqKey, entry);
  if (bundle.selectedPersona?.col) {
    marginalsCache.set(`${dsId}::${bundle.selectedPersona.col}`, entry);
  }
  return bundle;
}

export async function getQuestionMarginals(
  dsId: string,
  opts: QuestionMarginalsQuery
): Promise<QuestionMarginalsPayload> {
  const bundle = await loadOrRefreshMarginalsBundle(dsId, opts.persona);

  const siblingVs = matchQuestionVs(bundle.allVs, opts.questionId, undefined, opts.study);
  const matchedVs = opts.subitemId
    ? matchQuestionVs(bundle.allVs, opts.questionId, opts.subitemId, opts.study)
    : siblingVs;
  const exposedVs = siblingVs.length > 0 ? siblingVs : matchedVs;

  if (!bundle.selectedPersona) {
    return {
      dsId,
      found: false,
      reason: "No multi-level persona attributes available in this parquet.",
      parquetFile: bundle.parquetFile,
      loaderKind: bundle.loaderKind,
      availablePersonas: bundle.availablePersonas,
      selectedPersona: null,
      matchedVs: exposedVs,
      availableArms: [],
      selectedArm: null,
    };
  }

  const activeV = matchedVs[0];
  const qData = activeV ? bundle.questions[activeV] : undefined;
  if (!activeV || !qData) {
    return {
      dsId,
      found: false,
      reason: `Question ${JSON.stringify(opts.subitemId || opts.questionId)} is not present in ${bundle.parquetFile} (excluded or not yet rebuilt).`,
      parquetFile: bundle.parquetFile,
      loaderKind: bundle.loaderKind,
      availablePersonas: bundle.availablePersonas,
      selectedPersona: bundle.selectedPersona,
      matchedVs: exposedVs,
      availableArms: [],
      selectedArm: null,
    };
  }

  const availableArms = qData.arms;
  const selectedArm =
    opts.arm && availableArms.includes(opts.arm) ? opts.arm : null;
  const rawGroupCounts =
    qData.byArm[selectedArm ?? "__all__"] ?? {};

  const isMarginal = Boolean(bundle.isMarginal);
  const nOpts = qData.optionCodes.length;
  const overallCounts = new Array<number>(nOpts).fill(0);
  const groupKeysAll = Object.keys(rawGroupCounts);
  for (const cnts of Object.values(rawGroupCounts)) {
    for (let i = 0; i < nOpts; i += 1) {
      overallCounts[i] += cnts[i] ?? 0;
    }
  }
  const overallN = isMarginal
    ? groupKeysAll.length
    : overallCounts.reduce((a, b) => a + b, 0);
  if (overallN === 0) {
    return {
      dsId,
      found: false,
      reason: isMarginal
        ? `No subgroup marginal distributions found for ${JSON.stringify(activeV)} on ${bundle.selectedPersona.promptLabel} in ${bundle.parquetFile}.`
        : `No valid respondent answers found for ${JSON.stringify(activeV)} in ${bundle.parquetFile}.`,
      parquetFile: bundle.parquetFile,
      loaderKind: bundle.loaderKind,
      isMarginal,
      availablePersonas: bundle.availablePersonas,
      selectedPersona: bundle.selectedPersona,
      matchedVs: exposedVs,
      availableArms,
      selectedArm,
    };
  }

  const overallSum = overallCounts.reduce((a, b) => a + b, 0);
  const options: MarginalAnswerOption[] = qData.optionCodes.map((code, i) => ({
    code,
    text: qData.optionTexts[i],
    overallCount: isMarginal ? 0 : overallCounts[i],
    overallPct:
      overallSum > 0
        ? Math.round((overallCounts[i] / overallSum) * 10000) / 100
        : 0,
  }));

  const missingKey = "__MISSING__";
  const nonMissingKeys = Object.keys(rawGroupCounts).filter(
    (k) => k !== missingKey
  );
  const groupSum = (k: string) =>
    (rawGroupCounts[k] ?? []).reduce((a, b) => a + b, 0);

  if (bundle.canonicalLevelOrder) {
    const specOrder = new Map<string, number>();
    bundle.canonicalLevelOrder.forEach((lvl, idx) => specOrder.set(lvl, idx));
    nonMissingKeys.sort((a, b) => {
      const ia = specOrder.get(a) ?? 10000;
      const ib = specOrder.get(b) ?? 10000;
      if (ia !== ib) return ia - ib;
      if (!isMarginal) {
        const diffN = groupSum(b) - groupSum(a);
        if (diffN !== 0) return diffN;
      }
      return a.localeCompare(b);
    });
  } else {
    nonMissingKeys.sort((a, b) => {
      if (!isMarginal) {
        const diffN = groupSum(b) - groupSum(a);
        if (diffN !== 0) return diffN;
      }
      return a.localeCompare(b);
    });
  }

  const orderedKeys =
    missingKey in rawGroupCounts
      ? [...nonMissingKeys, missingKey]
      : nonMissingKeys;

  const isScore = qData.questionType === "score";
  const groups: MarginalGroupRow[] = orderedKeys.map((k) => {
    const cnts = rawGroupCounts[k];
    const gn = cnts.reduce((a, b) => a + b, 0);
    const shares = cnts.map((c) =>
      gn > 0 ? Math.round((c / gn) * 10000) / 100 : 0
    );
    let meanScore: number | null = null;
    if (isScore && gn > 0) {
      let weighted = 0;
      for (let i = 0; i < nOpts; i += 1) {
        weighted += (i + 1) * cnts[i];
      }
      meanScore = Math.round((weighted / gn) * 1000) / 1000;
    }
    return {
      level: k === missingKey ? "Missing (NA)" : k,
      isMissing: k === missingKey,
      n: isMarginal ? 0 : gn,
      counts: isMarginal ? new Array<number>(nOpts).fill(0) : cnts,
      shares,
      meanScore,
    };
  });

  let eligible = isMarginal
    ? groups.filter((g) => !g.isMissing)
    : groups.filter((g) => !g.isMissing && g.n >= 15);
  if (!isMarginal && eligible.length < 2) {
    eligible = groups.filter((g) => !g.isMissing && g.n > 0);
  }

  let maxTvdPct = 0;
  let tvdPair: [string, string] | null = null;
  let maxOptionDiff: MarginalSeparationStats["maxOptionDiff"] = null;
  let maxMeanDiff: MarginalSeparationStats["maxMeanDiff"] = null;

  if (eligible.length >= 2) {
    for (let i = 0; i < eligible.length; i += 1) {
      for (let j = i + 1; j < eligible.length; j += 1) {
        const ga = eligible[i];
        const gb = eligible[j];
        let l1 = 0;
        for (let oi = 0; oi < nOpts; oi += 1) {
          l1 += Math.abs(ga.shares[oi] - gb.shares[oi]);
        }
        const tvd = Math.round(0.5 * l1 * 100) / 100;
        if (tvd > maxTvdPct) {
          maxTvdPct = tvd;
          tvdPair = [ga.level, gb.level];
        }
        if (ga.meanScore !== null && gb.meanScore !== null) {
          const md = Math.round(Math.abs(ga.meanScore - gb.meanScore) * 1000) / 1000;
          if (maxMeanDiff === null || md > maxMeanDiff.delta) {
            const [hi, lo] =
              ga.meanScore >= gb.meanScore ? [ga, gb] : [gb, ga];
            maxMeanDiff = {
              delta: md,
              highGroup: hi.level,
              highMean: hi.meanScore!,
              lowGroup: lo.level,
              lowMean: lo.meanScore!,
            };
          }
        }
      }
    }

    for (let oi = 0; oi < nOpts; oi += 1) {
      let hi = eligible[0];
      let lo = eligible[0];
      for (const g of eligible) {
        if (g.shares[oi] > hi.shares[oi]) hi = g;
        if (g.shares[oi] < lo.shares[oi]) lo = g;
      }
      const delta = Math.round((hi.shares[oi] - lo.shares[oi]) * 100) / 100;
      if (maxOptionDiff === null || delta > maxOptionDiff.deltaPct) {
        maxOptionDiff = {
          option: qData.optionTexts[oi],
          deltaPct: delta,
          highGroup: hi.level,
          highPct: hi.shares[oi],
          lowGroup: lo.level,
          lowPct: lo.shares[oi],
        };
      }
    }
  }

  let overallMean: number | null = null;
  if (isScore && overallSum > 0) {
    let weighted = 0;
    for (let i = 0; i < nOpts; i += 1) {
      weighted += (i + 1) * overallCounts[i];
    }
    overallMean = Math.round((weighted / overallSum) * 1000) / 1000;
  }

  return {
    dsId,
    found: true,
    parquetFile: bundle.parquetFile,
    loaderKind: bundle.loaderKind,
    isMarginal,
    availablePersonas: bundle.availablePersonas,
    selectedPersona: bundle.selectedPersona,
    matchedVs: exposedVs,
    activeV,
    availableArms,
    selectedArm,
    questionType: qData.questionType,
    reversed: qData.reversed,
    overallN,
    overallMean,
    options,
    groups,
    separation: {
      minGroupN: isMarginal ? 0 : 15,
      eligibleGroupCount: eligible.length,
      maxTvdPct,
      tvdPair,
      maxOptionDiff,
      maxMeanDiff,
    },
  };
}
