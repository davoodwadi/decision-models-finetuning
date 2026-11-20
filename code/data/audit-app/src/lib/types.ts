export type SchemaKind = "question_audit" | "question_bank";

export type WorkspaceTab = "questions" | "personas";

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
  tab?: WorkspaceTab;
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

export interface PersonaLevelStat {
  value: string;
  count: number;
  pct: number;
  inGlobalLevels: boolean | null;
}

export interface PersonaAttributeSummary {
  col: string;
  promptLabel: string;
  source: "global" | "extra";
  isCountry: boolean;
  nPresent: number;
  nMissing: number;
  pctPresent: number;
  nLevels: number;
  levels: PersonaLevelStat[];
}

export interface PersonaRespondentRow {
  caseId: string;
  extraIds: Record<string, string | null>;
  nQuestions: number;
  nPresent: number;
  nPresentNoCountry: number;
  values: Record<string, string | null>;
  rawValues?: Record<string, string | null>;
}

export interface PersonaSummaryBundle {
  parquetFile: string;
  parquetMtimeMs: number;
  loaderKind: "build_persona_dataset" | "build_long_records";
  isspRuntimeMapped: boolean;
  nRespondents: number;
  nRows: number;
  nQuestions: number;
  extraIdCols: string[];
  hasCountry: boolean;
  nAttributes: number;
  uniquePersonasWithCountry: number;
  uniquePersonasWithoutCountry: number;
  avgAttrsPresent: number;
  minAttrsPresent: number;
  maxAttrsPresent: number;
  attributes: PersonaAttributeSummary[];
  respondents: PersonaRespondentRow[];
}

export type PersonaCompletenessFilter = "all" | "complete" | "incomplete";

export interface PersonasPagePayload {
  dsId: string;
  parquetFile: string;
  parquetMtimeMs: number;
  loaderKind: "build_persona_dataset" | "build_long_records";
  isspRuntimeMapped: boolean;
  nRespondents: number;
  nRows: number;
  nQuestions: number;
  extraIdCols: string[];
  hasCountry: boolean;
  nAttributes: number;
  uniquePersonasWithCountry: number;
  uniquePersonasWithoutCountry: number;
  avgAttrsPresent: number;
  minAttrsPresent: number;
  maxAttrsPresent: number;
  completeCount: number;
  incompleteCount: number;
  attributes: PersonaAttributeSummary[];
  totalFiltered: number;
  page: number;
  pageSize: number;
  totalPages: number;
  respondents: PersonaRespondentRow[];
}

export interface RespondentQuestionSample {
  qid: string;
  v: string;
  label: string | null;
  type: string;
  question: string;
  optionCodes: number[];
  optionTexts: string[];
  optionDescriptions: (string | null)[] | null;
  answerCode: number;
  answerText: string;
  answerIdx: number;
  arm: string | null;
}

export interface RespondentQuestionsPayload {
  dsId: string;
  caseId: string;
  totalQuestions: number;
  questions: RespondentQuestionSample[];
}
