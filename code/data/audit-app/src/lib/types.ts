export type SchemaKind = "question_audit" | "question_bank";

export type WorkspaceTab = "questions" | "personas" | "source";

export type ReviewFilter = "all" | "unreviewed" | "reviewed";

export type DecisionFilter = "all" | "included" | "excluded";

export type AuditFlag =
  | "persona_overlap"
  | "blocking"
  | "suspect"
  | "unsaved"
  | "overridden"
  | "tagged"
  | "option_relabeled"
  | "codebook_diff"
  | "text_changed"
  | "options_differ"
  | "not_in_source"
  | "identical";

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
  reviewFilter?: ReviewFilter;
  decisionFilter?: DecisionFilter;
  flags?: AuditFlag[];
  skip?: boolean;
  study?: string;
  families?: string[];
  categories?: string[];
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
  /** ISSP/WVS/HINTS: why the Parquet export would fail if this row were included as choice / as score (null = exportable). */
  export_problem_if_included?: { choice: string | null; score: string | null };
  /** ISSP/WVS/HINTS: valid answers per country (null = not a question column in the .dta). */
  n_valid_by_country?: Record<string, number | null>;
  /** True when this row is a respondent persona/demographic column rather than a survey question. */
  is_persona?: boolean;
  /** Canonical attribute name in persona_spec.yaml when this column feeds respondent personas (null otherwise). */
  persona_attribute?: string | null;
  /** ISSP/WVS/HINTS: true when option code-prefixing/relabeling modifies raw Stata option labels. */
  option_relabeled?: boolean;
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
  /** True when this column is automatically flagged as a respondent persona/demographic variable. */
  isPersona: boolean;
  /** Canonical attribute name in persona_spec.yaml if directly extracted as a persona attribute. */
  personaAttribute: string | null;
  /** True when this row is both a persona column and currently marked included (requires user fix). */
  personaConflict: boolean;
  isSuspect: boolean;
  suspectReasons: string[];
  /** Non-empty when this row is included but the Parquet export cannot handle it (computed, never saved). */
  exportProblem: string;
  /** Countries with zero valid answers for this question (informational; left out of that country's parquet). */
  notFieldedIn: string[];
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
  isMarginal?: boolean;
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
  isMarginal?: boolean;
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
  targetProbs?: number[];
}

export interface RespondentQuestionsPayload {
  dsId: string;
  caseId: string;
  totalQuestions: number;
  questions: RespondentQuestionSample[];
}

export interface MarginalPersonaOption {
  col: string;
  promptLabel: string;
  nLevels: number;
  nPresent: number;
}

export interface MarginalAnswerOption {
  code: number;
  text: string;
  overallCount: number;
  overallPct: number;
}

export interface MarginalGroupRow {
  level: string;
  isMissing: boolean;
  n: number;
  counts: number[];
  shares: number[];
  meanScore: number | null;
}

export interface MarginalSeparationStats {
  minGroupN: number;
  eligibleGroupCount: number;
  maxTvdPct: number;
  tvdPair: [string, string] | null;
  maxOptionDiff: {
    option: string;
    deltaPct: number;
    highGroup: string;
    highPct: number;
    lowGroup: string;
    lowPct: number;
  } | null;
  maxMeanDiff: {
    delta: number;
    highGroup: string;
    highMean: number;
    lowGroup: string;
    lowMean: number;
  } | null;
}

export interface QuestionMarginalsPayload {
  dsId: string;
  found: boolean;
  reason?: string;
  parquetFile: string;
  loaderKind: string;
  isMarginal?: boolean;
  availablePersonas: MarginalPersonaOption[];
  selectedPersona: MarginalPersonaOption | null;
  matchedVs: string[];
  activeV?: string;
  availableArms: string[];
  selectedArm: string | null;
  questionType?: string;
  reversed?: boolean;
  overallN?: number;
  overallMean?: number | null;
  options?: MarginalAnswerOption[];
  groups?: MarginalGroupRow[];
  separation?: MarginalSeparationStats;
}
