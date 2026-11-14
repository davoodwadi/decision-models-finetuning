export type SchemaKind = "question_audit" | "question_bank";

export type TriageMode =
  | "suspects"
  | "unsaved"
  | "overridden"
  | "tagged"
  | "unreviewed"
  | "substantive"
  | "included"
  | "excluded"
  | "all";

export interface QuestionOverride {
  stem?: string;
  subitems?: Record<string, string>;
  subitem_sep?: string;
  options?: Record<string, string>;
}

export interface InitialViewState {
  hasParams: boolean;
  rowKey?: string;
  mode?: TriageMode;
  skip?: boolean;
  study?: string;
  families?: string[];
  tags?: string[];
  q?: string;
}

export interface DatasetInfo {
  id: string;
  name: string;
  csvFile: string;
  path: string;
}

export interface OptionCount {
  option: string;
  count: number;
  description?: string | null;
}

export interface RowBreakdown {
  row: string;
  counts: number[];
}

export interface VariantItem {
  label: string;
  text: string;
  n: number;
}

export interface PairedVariant {
  n: number;
  stem: string;
  in_block_db_qid: string;
  in_block_db_text: string;
  qid_texts: Record<string, string>;
  options: OptionCount[];
  row_counts: number[][];
}

export interface SidebarSubItem {
  idx: number;
  id: string;
  label: string;
}

export interface ContextCandidate {
  qid: string;
  block: string;
  is_question: boolean;
  n_variants: number;
  preview: string;
  full_text: string;
}

export interface ItemDetail {
  QuestionType: string;
  Selector: string;
  rows: string[];
  options: OptionCount[];
  row_breakdowns: RowBreakdown[];
  qtext_variants: VariantItem[];
  db_variants: VariantItem[];
  opt_variant_sets: string[][];
  paired_variants?: PairedVariant[];
  paired_variants_total?: number;
  sample_texts: string[];
  in_block_db_text: string;
}

export interface AuditRow {
  rowIdx: number;
  rowKey: string;
  reviewed: boolean;
  hasOverride?: boolean;
  tags: string[];
  isDirty?: boolean;
  dirtyFields?: string[];
  isFormatSkipped: boolean;
  formatSkipLabel: string;
  formatSkipReason: string;
  isSuspect: boolean;
  suspectReasons: string[];
  data: Record<string, string>;
}

export interface DatasetPayload {
  dsId: string;
  csvPath: string;
  overridesPath: string;
  tagsPath: string;
  schemaKind: SchemaKind;
  columns: string[];
  hasNotebook: boolean;
  rows: AuditRow[];
  items: Record<string, ItemDetail>;
  studyContexts: Record<string, ContextCandidate[]>;
  overrides: Record<string, QuestionOverride>;
  tags: Record<string, string[]>;
}
