/** Shapes returned by scripts/source_check.py (codebook vs exported question comparison, Source tab). */

export type SourceTextStatus = "identical" | "changed" | "not_in_source";
export type SourceOptionsStatus = "match" | "differ" | "n/a";
export type SourceOptionRowStatus = "same" | "overridden" | "filled" | "dropped" | "final_only" | "source_only";
/** Where an exported option's text comes from (data/wvs_export.py OPTION_ORIGINS). */
export type SourceOptionOrigin = "stata_label" | "stata_label_code" | "stata_label_filled" | "code_only" | "override";

/** True when the exported wording or options differ from the codebook, or the codebook block could not be read. */
export function hasCodebookDiff(item: SourceCheckItem): boolean {
  return (
    item.error !== null ||
    item.text_status !== "identical" ||
    item.options_status === "differ"
  );
}

export interface SourceOptionRow {
  status: SourceOptionRowStatus;
  /** Codebook category matched to this row (null when there is none). */
  source: string | null;
  /** Exported option text (null for dropped / source_only rows). */
  final: string | null;
  /** Text before question_overrides.json (overridden) or the dropped option's text. */
  original: string | null;
  /** Provenance of the exported text (null for dropped / source_only rows). */
  origin: SourceOptionOrigin | null;
}

export interface SourceCodebookOption {
  text: string;
  missing_code: boolean;
}

export interface SourceCheckItem {
  v: string;
  included: boolean;
  /** True when this row is a respondent persona/demographic column rather than an exported survey question. */
  is_persona: boolean;
  type: string;
  export_problem: string | null;
  doc: string;
  page: number | null;
  source_text: string | null;
  final_text: string;
  final_options: string[];
  text_status: SourceTextStatus;
  text_diff: ["eq" | "del" | "ins", string][];
  source_options: SourceCodebookOption[];
  option_rows: SourceOptionRow[];
  options_status: SourceOptionsStatus;
  /** Review tags from question_tags.json for this row. */
  tags: string[];
  /** Stem-quality and review-tag warnings that require reviewer attention even when text matches the codebook. */
  warnings: string[];
  /** Why the codebook block could not be read for this question (null when it was read). */
  error: string | null;
}

export interface SourceCheckPayload {
  version: number;
  /** Revision token derived from comparison inputs; appended to PDF URLs so browser cache invalidates on edits. */
  revision: string;
  configured: boolean;
  documents: Record<string, { label: string; path: string }>;
  /** Where the exported option texts come from, e.g. the Stata file's value labels (null when not configured). */
  final_source: string | null;
  items: Record<string, SourceCheckItem>;
}
