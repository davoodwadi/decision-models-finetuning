import { execFile } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { discoverDatasets } from "./store";
import type { SourceCheckPayload } from "./sourceCheckTypes";

const execFileAsync = promisify(execFile);

const APP_DIR = process.cwd();
const DATA_DIR = path.resolve(APP_DIR, "..");
const CODE_DIR = path.resolve(DATA_DIR, "..");
const REPO_ROOT = path.resolve(CODE_DIR, "..");
const PYTHON_BIN = path.join(REPO_ROOT, ".venv-cu13", "bin", "python");
const SOURCE_CHECK_SCRIPT = path.join(APP_DIR, "scripts", "source_check.py");
/** Inputs of the comparison; the cached result is reused only while none of them changed. */
const INPUT_FILES = ["question_audit.csv", "question_overrides.json", "question_tags.json"];

type ComparisonPayload = Omit<SourceCheckPayload, "revision">;

const cache = new Map<string, { key: string; revision: string; payload: ComparisonPayload }>();
/** Highlighted codebook PDFs per row, keyed by dataset, row and comparison revision; oldest dropped first. */
const highlightCache = new Map<string, Buffer>();
const HIGHLIGHT_CACHE_MAX = 12;

async function datasetDirOf(dsId: string): Promise<string> {
  const { datasets } = await discoverDatasets();
  const match = datasets.find((d) => d.id === dsId);
  if (!match) {
    throw new Error(`Unknown dataset id: ${dsId}`);
  }
  return path.dirname(match.path);
}

async function mtimeOrMissing(p: string): Promise<string> {
  try {
    return String((await fs.stat(p)).mtimeMs);
  } catch (err: unknown) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return "missing";
    throw err;
  }
}

async function comparison(dsId: string, datasetDir: string): Promise<{ revision: string; payload: ComparisonPayload }> {
  // Exporter modules (code/data/*.py) decide the final text, so a change there also invalidates the cache.
  const exporterFiles = (await fs.readdir(DATA_DIR)).filter((f) => f.endsWith(".py")).sort();
  const stamps = await Promise.all([
    ...INPUT_FILES.map((f) => mtimeOrMissing(path.join(datasetDir, f))),
    ...exporterFiles.map((f) => mtimeOrMissing(path.join(DATA_DIR, f))),
    mtimeOrMissing(SOURCE_CHECK_SCRIPT),
  ]);
  const key = stamps.join("|");
  const hit = cache.get(dsId);
  if (hit && hit.key === key) return { revision: hit.revision, payload: hit.payload };

  const { stdout } = await execFileAsync(PYTHON_BIN, ["-W", "ignore", SOURCE_CHECK_SCRIPT, "--dataset-dir", datasetDir], {
    cwd: datasetDir,
    env: { ...process.env, PYTHONPATH: CODE_DIR },
    maxBuffer: 64 * 1024 * 1024,
  });
  const payload = JSON.parse(stdout) as ComparisonPayload;
  const revision = crypto.createHash("sha256").update(key).digest("hex").slice(0, 12);
  cache.set(dsId, { key, revision, payload });
  return { revision, payload };
}

export async function getSourceCheck(dsId: string): Promise<SourceCheckPayload> {
  const datasetDir = await datasetDirOf(dsId);
  const { revision, payload } = await comparison(dsId, datasetDir);
  return { ...payload, revision };
}

/** Absolute path of a codebook the dataset's provider lists; nothing else can be served. */
export async function getSourceDocumentPath(dsId: string, docId: string): Promise<string> {
  const datasetDir = await datasetDirOf(dsId);
  const { payload } = await comparison(dsId, datasetDir);
  const doc = payload.documents[docId];
  if (!doc) {
    throw new Error(`Unknown codebook document ${docId} for ${dsId}`);
  }
  const resolved = path.resolve(datasetDir, doc.path);
  if (!resolved.startsWith(datasetDir + path.sep) || !resolved.toLowerCase().endsWith(".pdf")) {
    throw new Error(`Codebook path must be a PDF inside the dataset folder: ${doc.path}`);
  }
  return resolved;
}

/** The row's codebook with its block highlighted (wording and each category by comparison status). */
export async function getHighlightedPdf(dsId: string, docId: string, rowKey: string): Promise<Buffer> {
  const datasetDir = await datasetDirOf(dsId);
  const { revision, payload } = await comparison(dsId, datasetDir);
  const item = payload.items[rowKey];
  if (!item) {
    throw new Error(`Unknown row key for ${dsId}: ${rowKey}`);
  }
  if (item.doc !== docId || item.page === null) {
    throw new Error(`${rowKey} has no block in codebook ${docId}`);
  }
  const cacheKey = `${dsId}|${rowKey}|${revision}`;
  const hit = highlightCache.get(cacheKey);
  if (hit) return hit;

  const statuses: Record<string, string> = {};
  for (const row of item.option_rows) {
    if (row.source !== null) statuses[row.source] = row.status;
  }
  const spec = JSON.stringify({ v: item.v, page: item.page, statuses });
  const { stdout } = await execFileAsync(
    PYTHON_BIN,
    ["-W", "ignore", SOURCE_CHECK_SCRIPT, "--dataset-dir", datasetDir, "--highlight-spec", spec],
    { cwd: datasetDir, env: { ...process.env, PYTHONPATH: CODE_DIR }, maxBuffer: 128 * 1024 * 1024, encoding: "buffer" }
  );
  highlightCache.set(cacheKey, stdout);
  while (highlightCache.size > HIGHLIGHT_CACHE_MAX) {
    const oldest = highlightCache.keys().next().value as string;
    highlightCache.delete(oldest);
  }
  return stdout;
}
