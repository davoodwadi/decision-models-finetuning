import type {
  AuditRow,
  ContextCandidate,
  ItemDetail,
  QuestionOverride,
  SchemaKind,
  SidebarSubItem,
} from "./types";

export const WARNING_TAGS = new Set([
  "has-dont-know",
  "odd-col-nums",
  "weird-col-nums",
  "prior-priming",
  "prior-task-dep",
  "html-table-flat",
  "attn-check",
  "empty-stem",
  "empty-options",
  "backref",
  "img-ref",
  "dup-qid",
  "short-text",
  "trunc-risk",
  "low-n",
  "unclean-text",
  "unclean text",
  "kev-type-check",
  "mpl-table-text",
  "rescuable-context",
  "in-grummon2025",
  "grummon2025-similar",
  "in-grummon2025-persona",
]);

const STANDARD_LIKERT_RE =
  /\b(strongly|agree|likely|unlikely|extremely|very|not at all|never|always|satisfied|dissatisfied|important|confident|accurate|trust|favor|oppose|support|willing|fair|acceptable|appropriate|often|frequently|rarely|much|little|extent|good|bad|slightly|moderately)\b/i;

const ORDINAL_CHOICE_RE =
  /\b(strongly|extremely|very|not at all|slightly|moderately)\b/i;

const FORMAT_SKIP_INFO: Record<string, { label: string; reason: string }> = {
  "0_DB": {
    label: "DB",
    reason: "Instruction / text display block (not a question)",
  },
  "0_TE": {
    label: "TEXT",
    reason: "Free-text entry item (non-discrete)",
  },
  "0_Slider": {
    label: "SLIDER",
    reason: "Continuous numeric slider (non-discrete)",
  },
  "0_CS": {
    label: "CONST-SUM",
    reason: "Constant-sum point allocation (non-discrete)",
  },
  "0_Consent_Check": {
    label: "CONSENT",
    reason: "Consent form / captcha / bot verification item",
  },
  "0_Demographics": {
    label: "DEMO",
    reason: "Demographic item already represented in respondent persona",
  },
  "5_MC_Multi": {
    label: "MULTI",
    reason: "Multi-select checkbox item (not a single categorical distribution)",
  },
};

export function makeRowKey(
  schemaKind: SchemaKind,
  columns: string[],
  data: Record<string, string>
): string {
  if (schemaKind === "question_audit") {
    if (columns.includes("study")) {
      return `${data.catalog_idx}:${data.study}:${data.QuestionID}`;
    }
    return `${data.catalog_idx}:${data.QuestionID}`;
  }
  return data.v;
}

export function getRawStem(
  schemaKind: SchemaKind,
  data: Record<string, string>
): string {
  return schemaKind === "question_audit"
    ? data.QuestionText ?? ""
    : data.question ?? "";
}

export function isNonEmptyOverride(
  ov: QuestionOverride | null | undefined
): ov is QuestionOverride {
  if (!ov) return false;
  if (ov.stem !== undefined) return true;
  if (ov.subitems && Object.keys(ov.subitems).length > 0) return true;
  if (ov.options && Object.keys(ov.options).length > 0) return true;
  return false;
}

export function normalizeQuestionOverride(
  ov: QuestionOverride | null | undefined,
  schemaKind: SchemaKind,
  data: Record<string, string>,
  detail: ItemDetail | undefined
): QuestionOverride | null {
  if (!ov) return null;
  const rawStem = getRawStem(schemaKind, data);
  const cleaned: QuestionOverride = {};

  if (ov.stem !== undefined && ov.stem !== rawStem) {
    cleaned.stem = ov.stem;
  }

  if (ov.subitems) {
    const subs: Record<string, string> = {};
    for (const k of Object.keys(ov.subitems).sort((a, b) => Number(a) - Number(b))) {
      const val = ov.subitems[k];
      const idx = Number(k);
      const rawSub =
        detail && Number.isInteger(idx) && idx >= 0 && idx < detail.rows.length
          ? detail.rows[idx]
          : undefined;
      if (val !== undefined && val !== rawSub) {
        subs[String(idx)] = val;
      }
    }
    if (Object.keys(subs).length > 0) {
      cleaned.subitems = subs;
    }
  }

  if (ov.options) {
    const opts: Record<string, string> = {};
    for (const orig of Object.keys(ov.options)) {
      const val = ov.options[orig]?.trim();
      if (val !== undefined && val !== "" && val !== orig) {
        opts[orig] = val;
      }
    }
    if (Object.keys(opts).length > 0) {
      cleaned.options = opts;
    }
  }

  return isNonEmptyOverride(cleaned) ? cleaned : null;
}

export function overridesEqual(
  a: QuestionOverride | null | undefined,
  b: QuestionOverride | null | undefined
): boolean {
  const na = isNonEmptyOverride(a) ? a : null;
  const nb = isNonEmptyOverride(b) ? b : null;
  if (!na && !nb) return true;
  if (!na || !nb) return false;
  if ((na.stem ?? null) !== (nb.stem ?? null)) return false;

  const subA = na.subitems ?? {};
  const subB = nb.subitems ?? {};
  const keysA = Object.keys(subA).sort();
  const keysB = Object.keys(subB).sort();
  if (keysA.length !== keysB.length) return false;
  for (let i = 0; i < keysA.length; i++) {
    if (keysA[i] !== keysB[i] || subA[keysA[i]] !== subB[keysB[i]]) return false;
  }

  const optA = na.options ?? {};
  const optB = nb.options ?? {};
  const oKeysA = Object.keys(optA).sort();
  const oKeysB = Object.keys(optB).sort();
  if (oKeysA.length !== oKeysB.length) return false;
  for (let i = 0; i < oKeysA.length; i++) {
    if (oKeysA[i] !== oKeysB[i] || optA[oKeysA[i]] !== optB[oKeysB[i]]) {
      return false;
    }
  }
  return true;
}

export function describeDirtyOverrideFields(
  savedOv: QuestionOverride | null | undefined,
  draftOv: QuestionOverride | null | undefined
): string[] {
  const s = isNonEmptyOverride(savedOv) ? savedOv : null;
  const d = isNonEmptyOverride(draftOv) ? draftOv : null;
  const fields: string[] = [];
  if ((s?.stem ?? null) !== (d?.stem ?? null)) {
    fields.push("override.stem");
  }
  if (JSON.stringify(s?.subitems ?? {}) !== JSON.stringify(d?.subitems ?? {})) {
    fields.push("override.subitems");
  }
  if (JSON.stringify(s?.options ?? {}) !== JSON.stringify(d?.options ?? {})) {
    fields.push("override.options");
  }
  return fields;
}

export function normalizeTag(raw: string): string {
  return raw.trim().replace(/^#+/, "").replace(/\s+/g, " ").trim();
}

export function normalizeTagList(
  tags: string[] | null | undefined
): string[] {
  if (!tags || !Array.isArray(tags)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of tags) {
    if (typeof raw !== "string") continue;
    const clean = normalizeTag(raw);
    if (!clean) continue;
    const key = clean.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(clean);
    }
  }
  return out;
}

export function tagsEqual(
  a: string[] | null | undefined,
  b: string[] | null | undefined
): boolean {
  const na = normalizeTagList(a);
  const nb = normalizeTagList(b);
  if (na.length !== nb.length) return false;
  return na.every((v, idx) => v === nb[idx]);
}

export function getSidebarSubItems(
  schemaKind: SchemaKind,
  data: Record<string, string>,
  detail: ItemDetail | undefined,
  override?: QuestionOverride
): SidebarSubItem[] {
  if (schemaKind !== "question_audit" || !detail) return [];

  if (detail.rows.length > 1) {
    const isBipolar =
      detail.QuestionType === "Matrix" && detail.Selector === "Bipolar";
    return detail.rows.map((origRow, idx) => {
      const rawRow = override?.subitems?.[String(idx)] ?? origRow;
      const id = `${data.QuestionID}_${idx + 1}`;
      if (isBipolar) {
        const colonIdx = rawRow.indexOf(":");
        if (colonIdx >= 0) {
          const left = rawRow.slice(0, colonIdx).trim();
          const right = rawRow.slice(colonIdx + 1).trim();
          return { idx, id, label: `${left} vs. ${right}` };
        }
      }
      return { idx, id, label: rawRow };
    });
  }

  const nSub = data.n_subitems ? Number(data.n_subitems) : 0;
  if (detail.rows.length === 0 && nSub > 1 && detail.qtext_variants.length > 1) {
    return detail.qtext_variants.map((v, idx) => ({
      idx,
      id: v.label,
      label: v.text,
    }));
  }

  return [];
}

const TERMINAL_PUNCT = new Set([".", "?", "!", ":", ";", "…", '"', "'", ")"]);

export function resolveContextText(
  contextSource: string,
  detail: ItemDetail | undefined,
  studyCandidates: ContextCandidate[] | undefined
): string {
  const cs = (contextSource || "").trim();
  if (!cs) return "";
  if (cs.startsWith("prefix:")) {
    return cs.slice("prefix:".length).trim();
  }
  if (cs === "in_block_db") {
    return detail ? detail.in_block_db_text : "";
  }
  if (cs.startsWith("qid:")) {
    if (!studyCandidates) return "";
    const byQid = new Map(studyCandidates.map((c) => [c.qid, c.full_text]));
    const parts: string[] = [];
    for (const rawQ of cs.slice("qid:".length).split("+")) {
      const q = rawQ.trim();
      const txt = byQid.get(q);
      if (txt) parts.push(txt);
    }
    return parts.join("\n\n");
  }
  return "";
}

export function decodeContextSep(rawSep: string | undefined): string {
  if (rawSep === undefined || rawSep === "") return "\n\n";
  return rawSep.replaceAll("\\n", "\n").replaceAll("\\t", "\t");
}

export function joinQuestionStem(
  resolvedCtx: string,
  questionText: string,
  firstMatrixRow?: string,
  rawContextSep?: string
): string {
  const sep = decodeContextSep(rawContextSep);
  let stem = "";
  if (resolvedCtx && questionText) {
    stem = `${resolvedCtx}${sep}${questionText}`;
  } else if (resolvedCtx) {
    stem = resolvedCtx;
  } else {
    stem = questionText;
  }

  if (firstMatrixRow) {
    if (!stem) return firstMatrixRow;
    if (
      !questionText &&
      resolvedCtx &&
      rawContextSep !== undefined &&
      rawContextSep !== ""
    ) {
      return `${resolvedCtx}${sep}${firstMatrixRow}`;
    }
    const lastChar = stem[stem.length - 1];
    return TERMINAL_PUNCT.has(lastChar)
      ? `${stem} ${firstMatrixRow}`
      : `${stem}\n\n${firstMatrixRow}`;
  }
  return stem.trim();
}

export interface ResolvedPairedVariant {
  vTag: string;
  n: number;
  ctxText: string;
  inBlockDbQid: string;
  inBlockDbText: string;
  rawStem: string;
  effectiveStem: string;
  options: Array<{
    rawOption: string;
    option: string;
    description?: string | null;
    isOverridden: boolean;
    count: number;
  }>;
  rowCounts: number[][];
  summaryHint: string;
}

function codeUnitCompare(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

function commonPrefixLen(strings: string[]): number {
  if (strings.length < 2) return 0;
  const first = strings[0];
  let len = first.length;
  for (let i = 1; i < strings.length; i++) {
    const s = strings[i];
    let j = 0;
    const maxJ = Math.min(len, s.length);
    while (j < maxJ && first.charCodeAt(j) === s.charCodeAt(j)) {
      j++;
    }
    len = j;
    if (len === 0) break;
  }
  return len;
}

function compactDiffSnippet(text: string, sharedPrefix: number, maxLen = 42): string {
  const clean = text.replace(/\s+/g, " ").trim();
  let start = 0;
  if (sharedPrefix >= 18 && sharedPrefix < clean.length) {
    const wordBack = clean.lastIndexOf(" ", sharedPrefix);
    start = wordBack >= 12 ? wordBack + 1 : sharedPrefix;
  }
  const sliced = start > 0 ? `…${clean.slice(start)}` : clean;
  return sliced.length > maxLen ? `${sliced.slice(0, maxLen - 1)}…` : sliced;
}

export function resolvePairedVariants(
  detail: ItemDetail | undefined,
  studyCandidates: ContextCandidate[] | undefined,
  effectiveContextSource: string,
  effectiveContextSep: string,
  effectiveOverride: QuestionOverride | undefined,
  dropSet: Set<string>
): ResolvedPairedVariant[] {
  if (!detail?.paired_variants || detail.paired_variants.length <= 1) {
    return [];
  }

  const cs = (effectiveContextSource || "").trim();
  const byQid = new Map((studyCandidates ?? []).map((c) => [c.qid, c.full_text]));

  interface MergedAcc {
    n: number;
    ctxText: string;
    inBlockDbQid: string;
    inBlockDbText: string;
    rawStem: string;
    effectiveStem: string;
    rawOptions: string[];
    rawDescriptions: Array<string | null | undefined>;
    counts: number[];
    rowCounts: number[][];
  }

  const mergedMap = new Map<string, MergedAcc>();

  for (const pv of detail.paired_variants) {
    let ctxText = "";
    if (cs.startsWith("prefix:")) {
      ctxText = cs.slice("prefix:".length).trim();
    } else if (cs === "in_block_db") {
      ctxText = pv.in_block_db_text || "";
    } else if (cs.startsWith("qid:")) {
      const parts: string[] = [];
      for (const rawQ of cs.slice("qid:".length).split("+")) {
        const q = rawQ.trim();
        if (!q) continue;
        const txt = pv.qid_texts?.[q] ?? byQid.get(q) ?? "";
        if (txt) parts.push(txt);
      }
      ctxText = parts.join("\n\n");
    }

    const rawStem = pv.stem;
    const effectiveStem =
      effectiveOverride?.stem !== undefined ? effectiveOverride.stem : rawStem;
    const rawOptions = pv.options.map((o) => o.option);
    const rawDescriptions = pv.options.map((o) => o.description);
    const mkey = `${ctxText}\u0000${effectiveStem}\u0000${rawOptions.join("\u0001")}\u0000${rawDescriptions.map((d) => d ?? "").join("\u0001")}`;

    const existing = mergedMap.get(mkey);
    if (!existing) {
      mergedMap.set(mkey, {
        n: pv.n,
        ctxText,
        inBlockDbQid: pv.in_block_db_qid,
        inBlockDbText: pv.in_block_db_text,
        rawStem,
        effectiveStem,
        rawOptions,
        rawDescriptions,
        counts: pv.options.map((o) => o.count),
        rowCounts: pv.row_counts.map((rc) => [...rc]),
      });
    } else {
      existing.n += pv.n;
      for (let i = 0; i < existing.counts.length && i < pv.options.length; i++) {
        existing.counts[i] += pv.options[i].count;
      }
      for (
        let r = 0;
        r < existing.rowCounts.length && r < pv.row_counts.length;
        r++
      ) {
        for (
          let c = 0;
          c < existing.rowCounts[r].length && c < pv.row_counts[r].length;
          c++
        ) {
          existing.rowCounts[r][c] += pv.row_counts[r][c];
        }
      }
    }
  }

  const merged = Array.from(mergedMap.values());
  if (merged.length <= 1) {
    return [];
  }

  const firstMatrixRow =
    detail.QuestionType === "Matrix" &&
    detail.Selector === "Likert" &&
    detail.rows.length > 0
      ? effectiveOverride?.subitems?.["0"] ?? detail.rows[0]
      : undefined;

  const decorated = merged.map((m) => {
    const options = m.rawOptions.map((rawOpt, idx) => ({
      rawOption: rawOpt,
      option: effectiveOverride?.options?.[rawOpt] ?? rawOpt,
      description: m.rawDescriptions[idx] ?? null,
      isOverridden: Boolean(
        effectiveOverride?.options && rawOpt in effectiveOverride.options
      ),
      count: m.counts[idx] ?? 0,
    }));
    const sortStem = joinQuestionStem(
      m.ctxText,
      m.effectiveStem,
      firstMatrixRow,
      effectiveContextSep
    );
    const activeOptLabels = options
      .filter((o) => !dropSet.has(o.rawOption))
      .map((o) => (o.description ? `${o.option}: ${o.description}` : o.option));
    return {
      ...m,
      options,
      sortStem,
      activeOptLabels,
    };
  });

  decorated.sort((a, b) => {
    const sc = codeUnitCompare(a.sortStem, b.sortStem);
    if (sc !== 0) return sc;
    const minLen = Math.min(a.activeOptLabels.length, b.activeOptLabels.length);
    for (let i = 0; i < minLen; i++) {
      const oc = codeUnitCompare(a.activeOptLabels[i], b.activeOptLabels[i]);
      if (oc !== 0) return oc;
    }
    return a.activeOptLabels.length - b.activeOptLabels.length;
  });

  const uniqueStems = new Set(
    decorated.map((d) => d.effectiveStem.replace(/\s+/g, " ").trim())
  );
  const uniqueCtxs = new Set(
    decorated.map((d) => d.ctxText.replace(/\s+/g, " ").trim())
  );
  const stemVaries = uniqueStems.size > 1;
  const ctxVaries = uniqueCtxs.size > 1;

  const allNamedArms =
    decorated.every(
      (d) => Boolean(d.inBlockDbQid) && !/^QID\d+/i.test(d.inBlockDbQid)
    ) &&
    new Set(decorated.map((d) => d.inBlockDbQid)).size === decorated.length;

  const cleanSources = decorated.map((d) =>
    (stemVaries
      ? d.effectiveStem
      : ctxVaries
        ? d.ctxText
        : d.options
            .map((o) => (o.description ? `${o.option}: ${o.description}` : o.option))
            .join(" | ")
    )
      .replace(/\s+/g, " ")
      .trim()
  );
  const sharedPref = commonPrefixLen(cleanSources);

  return decorated.map((d, idx) => ({
    vTag: `v${String(idx + 1).padStart(3, "0")}`,
    n: d.n,
    ctxText: d.ctxText,
    inBlockDbQid: d.inBlockDbQid,
    inBlockDbText: d.inBlockDbText,
    rawStem: d.rawStem,
    effectiveStem: d.effectiveStem,
    options: d.options,
    rowCounts: d.rowCounts,
    summaryHint: allNamedArms
      ? `${d.inBlockDbQid} · ${compactDiffSnippet(cleanSources[idx], sharedPref, 30)}`
      : compactDiffSnippet(cleanSources[idx], sharedPref),
  }));
}

export function isRowIncluded(schemaKind: SchemaKind, data: Record<string, string>): boolean {
  if (schemaKind === "question_audit") {
    return data.include === "True";
  }
  return data.type !== "exclude" && data.type !== "noul";
}

export function getEffectiveType(
  schemaKind: SchemaKind,
  data: Record<string, string>
): "score" | "choice" | "exclude" | "noul" {
  if (schemaKind === "question_audit") {
    if (data.include !== "True") return "exclude";
    return data.kev_type === "score" ? "score" : "choice";
  }
  if (data.type === "score" || data.type === "choice" || data.type === "noul") {
    return data.type;
  }
  return "exclude";
}

export function classifyAuditRow(
  rowIdx: number,
  schemaKind: SchemaKind,
  columns: string[],
  data: Record<string, string>,
  reviewed: boolean,
  detail: ItemDetail | undefined,
  override?: QuestionOverride,
  tags?: string[]
): AuditRow {
  const rowKey = makeRowKey(schemaKind, columns, data);
  const hasOverride = Boolean(isNonEmptyOverride(override));
  const cleanTags = normalizeTagList(tags);
  const warningTags = cleanTags.filter((t) =>
    WARNING_TAGS.has(t.toLowerCase())
  );

  let isFormatSkipped = false;
  let formatSkipLabel = "";
  let formatSkipReason = "";

  if (schemaKind === "question_audit") {
    const fam = data.family || "";
    if (fam in FORMAT_SKIP_INFO) {
      isFormatSkipped = true;
      formatSkipLabel = FORMAT_SKIP_INFO[fam].label;
      formatSkipReason = FORMAT_SKIP_INFO[fam].reason;
    } else if (fam.startsWith("0_")) {
      isFormatSkipped = true;
      formatSkipLabel = fam.slice(2).toUpperCase();
      formatSkipReason = `Format-excluded category (${fam})`;
    }
  } else {
    if (data.type === "noul") {
      isFormatSkipped = true;
      formatSkipLabel = "NOUL";
      formatSkipReason = "Unordered country-specific coding (not modeled)";
    }
  }

  const suspectReasons: string[] = [];
  if (!isFormatSkipped) {
    const dropSet = new Set(
      (data.drop_options || "")
        .split("|")
        .map((s) => s.trim())
        .filter(Boolean)
    );
    const rawOpts =
      detail && detail.options.length > 0
        ? detail.options.map((o) => o.option)
        : (data.Options_preview || data.options || "")
            .replace(/^\[\d+\]\s*/, "")
            .split(" | ")
            .map((s) => s.trim())
            .filter(Boolean);
    const activeOpts = rawOpts
      .filter((o) => !dropSet.has(o))
      .map((o) => override?.options?.[o] ?? o);
    const perVariantCounts =
      detail && detail.opt_variant_sets && detail.opt_variant_sets.length > 0
        ? detail.opt_variant_sets.map(
            (s) => s.filter((o) => !dropSet.has(o)).length
          )
        : [activeOpts.length];
    const nOpts = Math.max(...perVariantCounts, 0);
    const minOpts = Math.min(...perVariantCounts, nOpts);
    const optsLower = activeOpts.join(" | ").toLowerCase();

    for (const wt of warningTags) {
      const lower = wt.toLowerCase();
      if (lower === "has-dont-know" && dropSet.size > 0) {
        suspectReasons.push(`#${wt} (drop set)`);
      } else if (lower === "html-table-flat" && override?.stem !== undefined) {
        suspectReasons.push(`#${wt} (override set)`);
      } else {
        suspectReasons.push(`#${wt}`);
      }
    }

    if (schemaKind === "question_audit") {
      const inc = data.include === "True";
      const kt = data.kev_type;
      const fam = data.family || "";

      if (!inc) {
        suspectReasons.push("Excluded discrete");
      }
      if (fam === "1_Matrix_Bipolar") {
        suspectReasons.push("Bipolar scale");
      }
      if (inc && kt === "score") {
        if (nOpts > 0 && (minOpts <= 3 || nOpts >= 8)) {
          suspectReasons.push(`Score (${minOpts === nOpts ? nOpts : `${minOpts}–${nOpts}`} opts)`);
        } else if (
          fam === "4_MC_Single" &&
          nOpts > 0 &&
          !STANDARD_LIKERT_RE.test(optsLower)
        ) {
          suspectReasons.push("Check score anchors");
        }
      }
      if (
        inc &&
        kt === "choice" &&
        nOpts >= 5 &&
        ORDINAL_CHOICE_RE.test(optsLower)
      ) {
        suspectReasons.push("Choice looks ordinal");
      }
    } else {
      // question_bank schema
      if (data.type === "exclude") {
        suspectReasons.push("Excluded item");
      }
      if (data.country_dropped) {
        suspectReasons.push("Country options dropped");
      }
      if (data.type === "score" && nOpts > 0 && (nOpts <= 3 || nOpts >= 8)) {
        suspectReasons.push(`Score (${nOpts} opts)`);
      }
      if (
        data.type === "choice" &&
        nOpts >= 5 &&
        ORDINAL_CHOICE_RE.test(optsLower)
      ) {
        suspectReasons.push("Choice looks ordinal");
      }
    }
  }

  return {
    rowIdx,
    rowKey,
    reviewed,
    hasOverride,
    tags: cleanTags,
    isFormatSkipped,
    formatSkipLabel,
    formatSkipReason,
    isSuspect: suspectReasons.length > 0,
    suspectReasons,
    data,
  };
}
