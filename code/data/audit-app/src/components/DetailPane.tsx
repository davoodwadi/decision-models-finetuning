"use client";

import React, { useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Info,
  Layers,
  Pencil,
  RotateCcw,
  Save,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  defaultSubitemSep,
  getEffectiveType,
  getRawStem,
  isLegacyAutoSubitemSepDataset,
  joinQuestionStem,
  normalizeQuestionOverride,
  resolveContextText,
  resolvePairedVariants,
} from "@/lib/triage";
import type {
  AuditRow,
  ContextCandidate,
  ItemDetail,
  QuestionOverride,
  SchemaKind,
} from "@/lib/types";
import { TagEditor } from "./TagEditor";

interface DetailPaneProps {
  dsId: string;
  row: AuditRow | undefined;
  savedData: Record<string, string> | undefined;
  savedOverride?: QuestionOverride;
  effectiveOverride?: QuestionOverride;
  savedTags?: string[];
  allTagCounts?: Array<[string, number]>;
  detail: ItemDetail | undefined;
  studyCandidates: ContextCandidate[] | undefined;
  schemaKind: SchemaKind;
  columns: string[];
  positionIndex: number;
  totalFiltered: number;
  activeSubItemIdx: number;
  onSelectSubItemIdx: (idx: number) => void;
  onPrev: () => void;
  onNext: () => void;
  onStageRow: (rowIdx: number, patch: Record<string, string>) => void;
  onStageOverride?: (
    rowIdx: number,
    nextOverride: QuestionOverride | null
  ) => void;
  onStageTags?: (rowIdx: number, nextTags: string[]) => void;
  onFilterByTag?: (tag: string) => void;
  onSaveRow: (rowIdx: number) => void;
  onDiscardRow: (rowIdx: number) => void;
  onToggleReviewed: (rowIdx: number, nextReviewed: boolean) => void;
  onConfirmAndNext: () => void;
}

function initContextState(rawCsInput: string): {
  mode: "none" | "in_block" | "qid" | "prefix";
  qids: string[];
  prefix: string;
} {
  const rawCs = rawCsInput.trim();
  if (!rawCs) return { mode: "none", qids: [], prefix: "" };
  if (rawCs === "in_block_db") return { mode: "in_block", qids: [], prefix: "" };
  if (rawCs.startsWith("qid:")) {
    return {
      mode: "qid",
      qids: rawCs
        .slice(4)
        .split("+")
        .map((s) => s.trim())
        .filter(Boolean),
      prefix: "",
    };
  }
  if (rawCs.startsWith("prefix:")) {
    return {
      mode: "prefix",
      qids: [],
      prefix: rawCs.slice("prefix:".length).trim(),
    };
  }
  return { mode: "none", qids: [], prefix: "" };
}

export function DetailPane({
  dsId,
  row,
  savedData,
  savedOverride,
  effectiveOverride,
  savedTags = [],
  allTagCounts = [],
  detail,
  studyCandidates,
  schemaKind,
  columns,
  positionIndex,
  totalFiltered,
  activeSubItemIdx = 0,
  onSelectSubItemIdx,
  onPrev,
  onNext,
  onStageRow,
  onStageOverride,
  onStageTags,
  onFilterByTag,
  onSaveRow,
  onDiscardRow,
  onToggleReviewed,
  onConfirmAndNext,
}: DetailPaneProps) {
  const initialCtx = initContextState(row?.data.context_source ?? "");

  const [ctxMode, setCtxMode] = useState<"none" | "in_block" | "qid" | "prefix">(
    initialCtx.mode
  );
  const [selectedQids, setSelectedQids] = useState<string[]>(initialCtx.qids);
  const [prefixText, setPrefixText] = useState(initialCtx.prefix);
  const [sepInput, setSepInput] = useState(() => {
    const raw = row?.data.context_sep ?? "";
    return raw === "" ? "\\n\\n" : raw;
  });
  const [subitemSepInput, setSubitemSepInput] = useState(
    () => effectiveOverride?.subitem_sep ?? defaultSubitemSep(dsId)
  );
  const [blockSearch, setBlockSearch] = useState("");
  const [showStemCandidates, setShowStemCandidates] = useState(false);
  const [showFullSubItemPrompts, setShowFullSubItemPrompts] = useState(false);
  const [selectedVariantIdx, setSelectedVariantIdx] = useState(0);
  const [showPooledOptions, setShowPooledOptions] = useState(false);
  const [editingStem, setEditingStem] = useState(() =>
    Boolean(effectiveOverride && effectiveOverride.stem !== undefined)
  );
  const [showOriginalStem, setShowOriginalStem] = useState(false);
  const [editingSubItem, setEditingSubItem] = useState(false);
  const [editingOptionRaw, setEditingOptionRaw] = useState<string | null>(null);
  const [optionDraftText, setOptionDraftText] = useState("");
  const [noteInput, setNoteInput] = useState(() =>
    row
      ? schemaKind === "question_audit"
        ? row.data.review_note || ""
        : row.data.note || ""
      : ""
  );

  if (!row) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
        No questions match the current filter.
      </div>
    );
  }

  const d = row.data;
  const eff = getEffectiveType(schemaKind, d);

  const dropSet = new Set(
    (d.drop_options || "")
      .split("|")
      .map((s) => s.trim())
      .filter(Boolean)
  );

  const effectiveContextSource =
    ctxMode === "none"
      ? ""
      : ctxMode === "in_block"
        ? "in_block_db"
        : ctxMode === "qid"
          ? selectedQids.length > 0
            ? `qid:${selectedQids.join("+")}`
            : ""
          : prefixText.trim()
            ? `prefix:${prefixText.trim()}`
            : "";

  const effectiveContextSep =
    sepInput === "" || sepInput === "\\n\\n" ? "" : sepInput;

  const resolvedVariants =
    schemaKind === "question_audit" && columns.includes("context_source")
      ? resolvePairedVariants(
          detail,
          studyCandidates,
          effectiveContextSource,
          effectiveContextSep,
          effectiveOverride,
          dropSet,
          dsId
        )
      : [];
  const safeVariantIdx =
    resolvedVariants.length > 0
      ? Math.min(selectedVariantIdx, resolvedVariants.length - 1)
      : 0;
  const activePairedVariant =
    resolvedVariants.length > 0 ? resolvedVariants[safeVariantIdx] : undefined;

  const resolvedCtx = activePairedVariant
    ? activePairedVariant.ctxText
    : schemaKind === "question_audit"
      ? columns.includes("context_source")
        ? resolveContextText(effectiveContextSource, detail, studyCandidates)
        : d.Context_preview && !d.Context_preview.startsWith("[unused DB]")
          ? d.Context_preview
          : ""
      : "";

  const inBlockHintText =
    activePairedVariant?.inBlockDbText || detail?.in_block_db_text || "";
  const hasInBlockDb = Boolean(inBlockHintText);
  const topInBlockDbLabel =
    activePairedVariant?.inBlockDbQid ||
    (detail && detail.db_variants.length > 0 ? detail.db_variants[0].label : "");

  const csvRawQuestionStem = getRawStem(schemaKind, d);
  const rawQuestionStem = activePairedVariant
    ? activePairedVariant.rawStem
    : csvRawQuestionStem;
  const hasStemOverride = effectiveOverride?.stem !== undefined;
  const effectiveQuestionStem = hasStemOverride
    ? effectiveOverride?.stem ?? ""
    : rawQuestionStem;

  const stageOverrideUpdate = (updater: (cur: QuestionOverride) => QuestionOverride) => {
    if (!onStageOverride) return;
    const base: QuestionOverride = {
      ...(effectiveOverride?.stem !== undefined
        ? { stem: effectiveOverride.stem }
        : {}),
      ...(effectiveOverride?.subitems
        ? { subitems: { ...effectiveOverride.subitems } }
        : {}),
      ...(effectiveOverride?.subitem_sep !== undefined
        ? { subitem_sep: effectiveOverride.subitem_sep }
        : {}),
      ...(effectiveOverride?.options
        ? { options: { ...effectiveOverride.options } }
        : {}),
    };
    const updated = updater(base);
    const normalized = normalizeQuestionOverride(
      updated,
      schemaKind,
      d,
      detail,
      dsId
    );
    onStageOverride(row.rowIdx, normalized);
  };

  const pooledOptionItems = (
    detail && detail.options.length > 0
      ? detail.options
      : (d.Options_preview || d.options || "")
          .replace(/^\[\d+\]\s*/, "")
          .split(" | ")
          .filter(Boolean)
          .map((o) => ({ option: o.trim(), count: 0, description: null }))
  ).map((o) => ({
    rawOption: o.option,
    option: effectiveOverride?.options?.[o.option] ?? o.option,
    description: o.description ?? null,
    isOverridden: Boolean(
      effectiveOverride?.options && o.option in effectiveOverride.options
    ),
    count: o.count,
  }));

  const baseOptionItems =
    activePairedVariant && !showPooledOptions
      ? activePairedVariant.options
      : pooledOptionItems;

  const isMatrixLikert = Boolean(
    schemaKind === "question_audit" &&
      detail &&
      detail.QuestionType === "Matrix" &&
      detail.Selector === "Likert" &&
      detail.rows.length > 0
  );
  const isMultiRowBipolar = Boolean(
    schemaKind === "question_audit" &&
      detail &&
      detail.QuestionType === "Matrix" &&
      detail.Selector === "Bipolar" &&
      detail.rows.length > 1
  );
  const isMatrixWithSubItems = isMatrixLikert || isMultiRowBipolar;

  const matrixSubItems =
    isMatrixWithSubItems && detail
      ? detail.rows.map((origRowText, idx) => {
          const idxKey = String(idx);
          const isSubOverridden = Boolean(
            effectiveOverride?.subitems && idxKey in effectiveOverride.subitems
          );
          const rawRowText = isSubOverridden
            ? effectiveOverride?.subitems?.[idxKey] ?? origRowText
            : origRowText;
          const subItemId =
            detail.rows.length > 1
              ? `${d.QuestionID}_${idx + 1}`
              : d.QuestionID || `#${idx + 1}`;
          const bd = detail.row_breakdowns[idx];
          const activeRowCounts =
            activePairedVariant &&
            !showPooledOptions &&
            activePairedVariant.rowCounts[idx]
              ? activePairedVariant.rowCounts[idx]
              : bd
                ? bd.counts
                : [];
          if (isMultiRowBipolar) {
            const colonIdx = rawRowText.indexOf(":");
            const leftOpt =
              colonIdx >= 0 ? rawRowText.slice(0, colonIdx).trim() : rawRowText;
            const rightOpt =
              colonIdx >= 0 ? rawRowText.slice(colonIdx + 1).trim() : "";
            const subOpts = rightOpt
              ? [
                  {
                    rawOption: `1: ${leftOpt}`,
                    option: `1: ${leftOpt}`,
                    isOverridden: isSubOverridden,
                    count: activeRowCounts.length > 0 ? activeRowCounts[0] : 0,
                  },
                  {
                    rawOption: `2: ${rightOpt}`,
                    option: `2: ${rightOpt}`,
                    isOverridden: isSubOverridden,
                    count: activeRowCounts.length > 1 ? activeRowCounts[1] : 0,
                  },
                ]
              : baseOptionItems;
            return {
              idx,
              subItemId,
              origRowText,
              rawRowText,
              isSubOverridden,
              statementLabel: rightOpt
                ? `${leftOpt}  vs.  ${rightOpt}`
                : rawRowText,
              combinedStem: joinQuestionStem(
                resolvedCtx,
                effectiveQuestionStem,
                undefined,
                effectiveContextSep
              ),
              options: subOpts,
              totalN: subOpts.reduce((acc, o) => acc + o.count, 0),
            };
          }
          const subOpts = baseOptionItems.map((o, optIdx) => ({
            ...o,
            count: optIdx < activeRowCounts.length ? activeRowCounts[optIdx] : 0,
          }));
          return {
            idx,
            subItemId,
            origRowText,
            rawRowText,
            isSubOverridden,
            statementLabel: rawRowText,
            combinedStem: joinQuestionStem(
              resolvedCtx,
              effectiveQuestionStem,
              rawRowText,
              effectiveContextSep,
              effectiveOverride?.subitem_sep,
              dsId
            ),
            options: subOpts,
            totalN: subOpts.reduce((acc, o) => acc + o.count, 0),
          };
        })
      : [];

  const safeSubItemIdx =
    matrixSubItems.length > 0
      ? Math.min(activeSubItemIdx, matrixSubItems.length - 1)
      : 0;
  const activeSubItem = matrixSubItems[safeSubItemIdx];

  const isCollapsedLoop = Boolean(
    schemaKind === "question_audit" &&
      detail &&
      detail.rows.length === 0 &&
      d.n_subitems &&
      Number(d.n_subitems) > 1 &&
      detail.qtext_variants.length > 1
  );
  const safeLoopIdx =
    isCollapsedLoop && detail
      ? Math.min(activeSubItemIdx, detail.qtext_variants.length - 1)
      : 0;
  const activeLoopVariant =
    isCollapsedLoop && detail ? detail.qtext_variants[safeLoopIdx] : undefined;

  const fullStem = activeSubItem
    ? activeSubItem.combinedStem
    : activeLoopVariant
      ? joinQuestionStem(
          resolvedCtx,
          activeLoopVariant.text,
          undefined,
          effectiveContextSep
        )
      : schemaKind === "question_audit"
        ? joinQuestionStem(
            resolvedCtx,
            effectiveQuestionStem,
            undefined,
            effectiveContextSep
          )
        : effectiveQuestionStem.trim();

  const previewOptionItems =
    isMultiRowBipolar && activeSubItem
      ? activeSubItem.options
      : activePairedVariant
        ? activePairedVariant.options
        : baseOptionItems;

  const hasSubItemBreakdown = Boolean(
    activeSubItem && detail && detail.row_breakdowns.length > safeSubItemIdx
  );

  const optionItems =
    (isMultiRowBipolar || hasSubItemBreakdown) && activeSubItem
      ? activeSubItem.options
      : baseOptionItems;

  const totalCount = optionItems.reduce((acc, o) => acc + o.count, 0);

  const handleSetDecision = (target: "score" | "choice" | "exclude" | "noul") => {
    if (schemaKind === "question_audit") {
      if (target === "exclude") {
        onStageRow(row.rowIdx, { include: "False", kev_type: "" });
      } else if (target === "score" || target === "choice") {
        onStageRow(row.rowIdx, { include: "True", kev_type: target });
      }
    } else {
      onStageRow(row.rowIdx, { type: target });
    }
  };

  const toggleDropOption = (optLabel: string) => {
    const next = new Set(dropSet);
    if (next.has(optLabel)) {
      next.delete(optLabel);
    } else {
      next.add(optLabel);
    }
    const joined = Array.from(next).join(" | ");
    onStageRow(row.rowIdx, { drop_options: joined });
  };

  const clearAllDrops = () => {
    onStageRow(row.rowIdx, { drop_options: "" });
  };

  const stageContextSource = (nextCs: string) => {
    onStageRow(row.rowIdx, { context_source: nextCs.trim() });
  };

  const handleSelectContextMode = (
    nextMode: "none" | "in_block" | "qid" | "prefix"
  ) => {
    setCtxMode(nextMode);
    if (nextMode === "none") {
      setSelectedQids([]);
      setPrefixText("");
      stageContextSource("");
    } else if (nextMode === "in_block") {
      setSelectedQids([]);
      setPrefixText("");
      stageContextSource("in_block_db");
    } else if (nextMode === "qid") {
      if (selectedQids.length === 0 && topInBlockDbLabel) {
        const init = [topInBlockDbLabel];
        setSelectedQids(init);
        stageContextSource(`qid:${init.join("+")}`);
      } else {
        stageContextSource(
          selectedQids.length > 0 ? `qid:${selectedQids.join("+")}` : ""
        );
      }
    } else if (nextMode === "prefix") {
      const trimmed = prefixText.trim();
      stageContextSource(trimmed ? `prefix:${trimmed}` : "");
    }
  };

  const handleToggleCandidateQid = (qid: string) => {
    const next = selectedQids.includes(qid)
      ? selectedQids.filter((x) => x !== qid)
      : [...selectedQids, qid];
    setSelectedQids(next);
    stageContextSource(next.length > 0 ? `qid:${next.join("+")}` : "");
  };

  const isDirty = Boolean(row.isDirty);
  const dirtyFields = row.dirtyFields ?? [];
  const noteField = schemaKind === "question_audit" ? "review_note" : "note";

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-50">
      {/* Sticky Top Action Bar */}
      <div className="bg-white border-b border-slate-200 px-5 py-2.5 flex items-center gap-3 shrink-0 flex-wrap">
        <div className="flex items-center gap-1.5">
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            onClick={onPrev}
            disabled={positionIndex <= 0}
            className="bg-white cursor-pointer"
            title="Previous (k / ↑)"
          >
            <ChevronLeft className="w-4 h-4 text-slate-700" />
          </Button>
          <span className="text-xs font-mono text-slate-500 min-w-[68px] text-center">
            {positionIndex + 1} / {totalFiltered}
          </span>
          <Button
            type="button"
            variant="outline"
            size="icon-sm"
            onClick={onNext}
            disabled={positionIndex >= totalFiltered - 1}
            className="bg-white cursor-pointer"
            title="Next (j / ↓)"
          >
            <ChevronRight className="w-4 h-4 text-slate-700" />
          </Button>
        </div>

        <div className="h-5 w-px bg-slate-200" />

        {/* Modeling Type Segmented Buttons */}
        <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-100/80">
          <button
            type="button"
            onClick={() => handleSetDecision("score")}
            className={[
              "px-3 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer",
              eff === "score"
                ? "bg-violet-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            Score <kbd className="ml-1 opacity-75 font-mono text-[10px]">s</kbd>
          </button>
          <button
            type="button"
            onClick={() => handleSetDecision("choice")}
            className={[
              "px-3 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer",
              eff === "choice"
                ? "bg-blue-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            Choice <kbd className="ml-1 opacity-75 font-mono text-[10px]">c</kbd>
          </button>
          {schemaKind === "question_bank" && (
            <button
              type="button"
              onClick={() => handleSetDecision("noul")}
              className={[
                "px-3 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer",
                eff === "noul"
                  ? "bg-slate-600 text-white shadow-xs"
                  : "text-slate-600 hover:text-slate-900",
              ].join(" ")}
            >
              Noul
            </button>
          )}
          <button
            type="button"
            onClick={() => handleSetDecision("exclude")}
            className={[
              "px-3 py-1 text-xs font-semibold rounded-md transition-all cursor-pointer",
              eff === "exclude"
                ? row.isFormatSkipped
                  ? "bg-slate-500 text-white shadow-xs"
                  : "bg-rose-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            Exclude <kbd className="ml-1 opacity-75 font-mono text-[10px]">x</kbd>
          </button>
        </div>

        {schemaKind === "question_bank" && columns.includes("reversed") && (
          <label className="inline-flex items-center gap-1.5 text-xs text-slate-700 cursor-pointer select-none ml-1">
            <Checkbox
              checked={d.reversed === "True"}
              onCheckedChange={(checked) =>
                onStageRow(row.rowIdx, {
                  reversed: checked ? "True" : "False",
                })
              }
              className="bg-white"
            />
            Reversed scale
          </label>
        )}

        <div className="flex-1" />

        {/* Unsaved Draft Controls, Review Status & Confirm */}
        {isDirty && (
          <>
            <Badge
              variant="outline"
              className="rounded-md text-[11px] font-bold bg-orange-100 text-orange-900 border-orange-300 uppercase tracking-tight"
            >
              Unsaved ({dirtyFields.length})
            </Badge>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onDiscardRow(row.rowIdx)}
              className="bg-white text-slate-600 hover:bg-slate-50 text-xs cursor-pointer"
              title="Discard unsaved changes on this question (Esc)"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Discard
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => onSaveRow(row.rowIdx)}
              className="border-emerald-300 bg-emerald-50 text-emerald-900 hover:bg-emerald-100 text-xs cursor-pointer"
              title="Save changes to CSV without advancing (Ctrl/Cmd+S)"
            >
              <Save className="w-3.5 h-3.5 text-emerald-700" />
              Save
            </Button>
          </>
        )}

        {!isDirty && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onToggleReviewed(row.rowIdx, !row.reviewed)}
            className={[
              "text-xs cursor-pointer",
              row.reviewed
                ? "bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100"
                : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50",
            ].join(" ")}
            title="Toggle reviewed without modifying CSV (Space)"
          >
            <CheckCircle2
              className={`w-3.5 h-3.5 ${row.reviewed ? "text-emerald-600" : "text-slate-400"}`}
            />
            {row.reviewed ? "Reviewed" : "Mark reviewed"}
          </Button>
        )}

        <Button
          type="button"
          size="sm"
          onClick={onConfirmAndNext}
          className={[
            "text-xs font-semibold text-white cursor-pointer",
            isDirty
              ? "bg-emerald-600 hover:bg-emerald-700"
              : "bg-slate-900 hover:bg-slate-800",
          ].join(" ")}
          title={
            isDirty
              ? "Write staged changes to CSV, mark reviewed, and advance (Enter)"
              : "Mark reviewed and advance (Enter)"
          }
        >
          <Check className="w-3.5 h-3.5" />
          {isDirty ? "Save, Confirm & Next" : "Confirm & Next"}
          <kbd
            className={[
              "ml-1 px-1 py-0.2 rounded text-[10px] font-mono",
              isDirty ? "bg-emerald-800" : "bg-slate-700",
            ].join(" ")}
          >
            ↵
          </kbd>
        </Button>
      </div>

      {/* Scrollable Content */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Question Metadata Header Card */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs space-y-3">
          {isDirty && savedData && (
            <div className="flex items-center justify-between gap-3 flex-wrap px-3 py-2 rounded-lg bg-orange-50 border border-orange-200 text-orange-950 text-xs">
              <div className="flex items-center gap-2 flex-wrap min-w-0">
                <span className="font-bold text-orange-900 uppercase text-[11px] tracking-tight">
                  Unsaved draft (not saved yet):
                </span>
                {dirtyFields.map((col) => {
                  if (col === "tags") {
                    return (
                      <span
                        key={col}
                        className="px-2 py-0.5 rounded bg-white border border-orange-200 font-mono text-[11px] text-slate-800 truncate max-w-md"
                      >
                        <strong>tags</strong>:{" "}
                        <span className="text-slate-400">
                          [{savedTags.map((t) => `#${t}`).join(", ")}]
                        </span>{" "}
                        →{" "}
                        <span className="text-orange-800 font-semibold">
                          [{(row.tags ?? []).map((t) => `#${t}`).join(", ")}]{" "}
                          (question_tags.json)
                        </span>
                      </span>
                    );
                  }
                  if (col.startsWith("override.")) {
                    const label =
                      col === "override.stem"
                        ? savedOverride?.stem !== undefined &&
                          effectiveOverride?.stem === undefined
                          ? "reverted to raw stem"
                          : "modified"
                        : col === "override.subitems"
                          ? `${Object.keys(effectiveOverride?.subitems ?? {}).length} sub-item(s)`
                          : col === "override.subitem_sep"
                            ? `delimiter ${JSON.stringify(effectiveOverride?.subitem_sep ?? defaultSubitemSep(dsId))}`
                            : `${Object.keys(effectiveOverride?.options ?? {}).length} option(s)`;
                    return (
                      <span
                        key={col}
                        className="px-2 py-0.5 rounded bg-white border border-orange-200 font-mono text-[11px] text-slate-800 truncate max-w-md"
                      >
                        <strong>{col}</strong>:{" "}
                        <span className="text-orange-800 font-semibold">
                          {label} (question_overrides.json)
                        </span>
                      </span>
                    );
                  }
                  return (
                    <span
                      key={col}
                      className="px-2 py-0.5 rounded bg-white border border-orange-200 font-mono text-[11px] text-slate-800 truncate max-w-md"
                    >
                      <strong>{col}</strong>:{" "}
                      <span className="text-slate-400">
                        {JSON.stringify(savedData[col] ?? "")}
                      </span>{" "}
                      →{" "}
                      <span className="text-orange-800 font-semibold">
                        {JSON.stringify(d[col] ?? "")}
                      </span>
                    </span>
                  );
                })}
              </div>
              <span className="text-[11px] text-orange-800 shrink-0">
                Click <strong>Save, Confirm &amp; Next</strong> (↵) to commit
              </span>
            </div>
          )}
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-base font-bold font-mono text-slate-900">
                  {schemaKind === "question_audit"
                    ? `${d.study ? `${d.study} · ` : ""}${d.QuestionID}`
                    : `${d.v}${d.source_q ? ` (${d.source_q})` : ""}`}
                </h2>
                {schemaKind === "question_audit" && (
                  <>
                    <span className="px-2 py-0.5 text-xs font-mono rounded bg-slate-100 text-slate-700">
                      {d.family}
                    </span>
                    {detail?.QuestionType && (
                      <span className="px-2 py-0.5 text-xs font-mono rounded bg-slate-100 text-slate-600">
                        {detail.QuestionType}
                        {detail.Selector ? `/${detail.Selector}` : ""}
                      </span>
                    )}
                    {d.BlockName && (
                      <span className="text-xs text-slate-500">
                        Block: <strong className="text-slate-700">{d.BlockName}</strong>
                      </span>
                    )}
                    {d.n_answered && (
                      <span className="text-xs text-slate-500">
                        N={d.n_answered}
                        {d.n_subitems && Number(d.n_subitems) > 1
                          ? ` (${d.n_subitems} sub-items)`
                          : ""}
                      </span>
                    )}
                  </>
                )}
                {schemaKind === "question_bank" && d.data_label && (
                  <span className="text-xs text-slate-500">{d.data_label}</span>
                )}
              </div>
            </div>
            <span className="text-[11px] font-mono text-slate-400">
              row #{row.rowIdx}
            </span>
          </div>

          {onStageTags && (
            <TagEditor
              tags={row.tags ?? []}
              allTagCounts={allTagCounts}
              onChange={(nextTags) => onStageTags(row.rowIdx, nextTags)}
              onFilterByTag={onFilterByTag}
            />
          )}

          {/* Format-skipped banner (greyed-out explanation) */}
          {row.isFormatSkipped && eff !== "score" && eff !== "choice" && (
            <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-xs">
              <Info className="w-4 h-4 text-slate-400 shrink-0" />
              <span>
                <strong>Auto-skipped by format ({row.formatSkipLabel}):</strong>{" "}
                {row.formatSkipReason}. Kept in catalog only for survey flow &amp; context lookup.
              </span>
            </div>
          )}

          {/* Suspect reasons banner */}
          {row.isSuspect && (
            <div className="flex items-center gap-2 flex-wrap px-3 py-2 rounded-lg bg-amber-50/90 border border-amber-200 text-amber-900 text-xs">
              <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
              <span className="font-semibold">Why flagged as suspect:</span>
              {row.suspectReasons.map((r) => (
                <span
                  key={r}
                  className="px-2 py-0.5 rounded-full bg-amber-100/90 border border-amber-300/80 font-medium text-[11px]"
                >
                  {r}
                </span>
              ))}
            </div>
          )}

          {/* Question Stem & Prepended Context Controls */}
          <div className="space-y-2.5 pt-1">
            {resolvedVariants.length > 1 && activePairedVariant && (
              <div className="rounded-lg border border-indigo-200/90 bg-indigo-50/45 p-2.5 space-y-2">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <Layers className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
                    <span className="text-[11px] font-bold uppercase tracking-wider text-indigo-900">
                      Paired Experimental Variants ({resolvedVariants.length})
                    </span>
                    <span className="text-[11px] text-indigo-700/80">
                      — context, stem &amp; options stay paired per respondent arm (
                      <code className="font-mono text-[10px]">
                        _{resolvedVariants[0].vTag}…_
                        {resolvedVariants[resolvedVariants.length - 1].vTag}
                      </code>
                      )
                    </span>
                  </div>
                  <div className="inline-flex items-center gap-1 rounded-md bg-white border border-indigo-200 px-1.5 py-0.5 text-xs">
                    <button
                      type="button"
                      disabled={safeVariantIdx <= 0}
                      onClick={() =>
                        setSelectedVariantIdx(Math.max(0, safeVariantIdx - 1))
                      }
                      className="px-1 text-indigo-700 hover:text-indigo-950 disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                      title="Previous variant"
                    >
                      ◀
                    </button>
                    <span className="font-mono text-[11px] font-semibold text-indigo-900 px-1">
                      {activePairedVariant.vTag} ({safeVariantIdx + 1}/
                      {resolvedVariants.length} · n={activePairedVariant.n})
                    </span>
                    <button
                      type="button"
                      disabled={safeVariantIdx >= resolvedVariants.length - 1}
                      onClick={() =>
                        setSelectedVariantIdx(
                          Math.min(
                            resolvedVariants.length - 1,
                            safeVariantIdx + 1
                          )
                        )
                      }
                      className="px-1 text-indigo-700 hover:text-indigo-950 disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                      title="Next variant"
                    >
                      ▶
                    </button>
                  </div>
                </div>

                {resolvedVariants.length <= 12 ? (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {resolvedVariants.map((rv, idx) => {
                      const active = idx === safeVariantIdx;
                      return (
                        <button
                          key={rv.vTag}
                          type="button"
                          onClick={() => setSelectedVariantIdx(idx)}
                          title={`${rv.vTag} (n=${rv.n}): ${rv.summaryHint}`}
                          className={[
                            "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs transition-colors cursor-pointer max-w-xs",
                            active
                              ? "bg-indigo-600 border-indigo-600 text-white shadow-2xs font-medium"
                              : "bg-white border-indigo-200/80 text-slate-700 hover:bg-indigo-50/70",
                          ].join(" ")}
                        >
                          <span className="font-mono text-[11px] font-semibold shrink-0">
                            {rv.vTag}
                          </span>
                          <span
                            className={[
                              "font-mono text-[10px] shrink-0",
                              active ? "text-indigo-100" : "text-slate-400",
                            ].join(" ")}
                          >
                            n={rv.n}
                          </span>
                          {rv.summaryHint && (
                            <span
                              className={[
                                "truncate text-[11px]",
                                active ? "text-indigo-50" : "text-slate-500",
                              ].join(" ")}
                            >
                              {rv.summaryHint}
                            </span>
                          )}
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <select
                    value={safeVariantIdx}
                    onChange={(e) =>
                      setSelectedVariantIdx(Number(e.target.value))
                    }
                    className="h-7 w-full rounded-md border border-indigo-200 bg-white px-2 text-xs text-slate-800 font-mono"
                  >
                    {resolvedVariants.map((rv, idx) => (
                      <option key={rv.vTag} value={idx}>
                        {rv.vTag} (n={rv.n}) — {rv.summaryHint}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}

            {columns.includes("context_source") && !row.isFormatSkipped && (
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="text-xs font-semibold text-slate-600">
                    Extra context glued before question:
                  </span>
                  <div className="inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-100/80 flex-wrap">
                    <button
                      type="button"
                      onClick={() => handleSelectContextMode("none")}
                      className={[
                        "px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer",
                        ctxMode === "none"
                          ? "bg-white text-slate-900 shadow-2xs font-semibold"
                          : "text-slate-600 hover:text-slate-900",
                      ].join(" ")}
                    >
                      No extra context
                    </button>
                    <button
                      type="button"
                      disabled={!hasInBlockDb}
                      onClick={() => handleSelectContextMode("in_block")}
                      title={
                        hasInBlockDb
                          ? "Prepend the instruction/stimulus block shown directly above this question"
                          : "No preceding text block exists in this Qualtrics block"
                      }
                      className={[
                        "px-2.5 py-1 text-xs font-medium rounded-md transition-colors",
                        !hasInBlockDb
                          ? "opacity-40 cursor-not-allowed text-slate-400"
                          : ctxMode === "in_block"
                            ? "bg-blue-600 text-white shadow-2xs font-semibold cursor-pointer"
                            : "text-slate-600 hover:text-slate-900 cursor-pointer",
                      ].join(" ")}
                    >
                      Stimulus above in block
                      {topInBlockDbLabel ? ` (${topInBlockDbLabel})` : ""}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSelectContextMode("qid")}
                      className={[
                        "px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer",
                        ctxMode === "qid"
                          ? "bg-blue-600 text-white shadow-2xs font-semibold"
                          : "text-slate-600 hover:text-slate-900",
                      ].join(" ")}
                    >
                      Pick block(s) from study
                      {ctxMode === "qid" && selectedQids.length > 0
                        ? ` (${selectedQids.length})`
                        : ""}
                    </button>
                    <button
                      type="button"
                      onClick={() => handleSelectContextMode("prefix")}
                      className={[
                        "px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer",
                        ctxMode === "prefix"
                          ? "bg-blue-600 text-white shadow-2xs font-semibold"
                          : "text-slate-600 hover:text-slate-900",
                      ].join(" ")}
                    >
                      Custom prefix
                    </button>
                  </div>
                </div>

                {/* Picker when "Pick block(s) from study" is selected */}
                {ctxMode === "qid" && studyCandidates && (
                  <div className="p-3 rounded-lg bg-slate-50 border border-slate-200 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <Input
                        type="text"
                        value={blockSearch}
                        onChange={(e) => setBlockSearch(e.target.value)}
                        placeholder="Search text blocks in this study..."
                        className="h-7 flex-1 text-xs bg-white shadow-none"
                      />
                      <label className="inline-flex items-center gap-1.5 text-[11px] text-slate-600 cursor-pointer select-none">
                        <Checkbox
                          checked={showStemCandidates}
                          onCheckedChange={(checked) =>
                            setShowStemCandidates(Boolean(checked))
                          }
                          className="bg-white"
                        />
                        Include other question stems
                      </label>
                    </div>
                    {selectedQids.length > 0 && (
                      <div className="text-[11px] text-blue-800 font-medium">
                        Selected order: {selectedQids.join(" → ")}
                      </div>
                    )}
                    <div className="max-h-48 overflow-y-auto space-y-1 pr-1">
                      {studyCandidates
                        .filter((c) => {
                          if (!showStemCandidates && c.is_question && !selectedQids.includes(c.qid)) {
                            return false;
                          }
                          if (!blockSearch.trim()) return true;
                          const q = blockSearch.toLowerCase();
                          return (
                            c.qid.toLowerCase().includes(q) ||
                            c.block.toLowerCase().includes(q) ||
                            c.full_text.toLowerCase().includes(q)
                          );
                        })
                        .map((cand) => {
                          const checked = selectedQids.includes(cand.qid);
                          return (
                            <label
                              key={cand.qid}
                              className={[
                                "flex items-start gap-2 p-2 rounded border text-xs cursor-pointer transition-colors",
                                checked
                                  ? "bg-blue-50 border-blue-300 text-slate-900"
                                  : "bg-white border-slate-200/80 text-slate-700 hover:bg-slate-50",
                              ].join(" ")}
                            >
                              <Checkbox
                                checked={checked}
                                onCheckedChange={() =>
                                  handleToggleCandidateQid(cand.qid)
                                }
                                className="mt-0.5 bg-white"
                              />
                              <div className="min-w-0 flex-1">
                                <div className="flex items-center gap-2 text-[11px] text-slate-500 mb-0.5">
                                  <span className="font-mono font-semibold text-slate-700">
                                    {cand.qid}
                                  </span>
                                  <span>
                                    {cand.is_question
                                      ? cand.block
                                        ? `Question stem · ${cand.block}`
                                        : "Question stem"
                                      : `Block: ${cand.block || "Instruction"}`}
                                  </span>
                                </div>
                                <div className="line-clamp-2 text-xs text-slate-700">
                                  {cand.full_text}
                                </div>
                              </div>
                            </label>
                          );
                        })}
                    </div>
                  </div>
                )}

                {/* Input when "Custom prefix" is selected */}
                {ctxMode === "prefix" && (
                  <div className="flex items-center gap-2">
                    <Input
                      type="text"
                      value={prefixText}
                      onChange={(e) => {
                        const rawVal = e.target.value;
                        setPrefixText(rawVal);
                        const trimmed = rawVal.trim();
                        stageContextSource(trimmed ? `prefix:${trimmed}` : "");
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          e.stopPropagation();
                          (e.target as HTMLInputElement).blur();
                        }
                      }}
                      placeholder="Type custom sentence to prepend (e.g. Please indicate how much you agree with:)"
                      className="h-8 w-full text-xs bg-white shadow-none"
                    />
                  </div>
                )}
              </div>
            )}

            {/* Active Prepended Context Box */}
            {resolvedCtx && (
              <div className="p-3 rounded-lg bg-blue-50/60 border border-blue-100 text-xs text-slate-700 leading-relaxed whitespace-pre-wrap space-y-2">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-blue-700">
                    Prepended Context
                    {activePairedVariant
                      ? ` (${activePairedVariant.vTag})`
                      : ""}
                    {ctxMode === "in_block"
                      ? ` — Stimulus above in block${topInBlockDbLabel ? ` (${topInBlockDbLabel})` : ""}`
                      : ctxMode === "qid"
                        ? ` — Survey block (${selectedQids.join(" + ")})`
                        : ctxMode === "prefix"
                          ? " — Custom prefix"
                          : ""}
                  </span>
                  {columns.includes("context_sep") && (
                    <div className="flex items-center gap-1.5">
                      <label
                        htmlFor="context-sep-input"
                        className="text-[11px] font-medium text-slate-600"
                      >
                        Delimiter:
                      </label>
                      <Input
                        id="context-sep-input"
                        type="text"
                        value={sepInput}
                        onChange={(e) => {
                          const v = e.target.value;
                          setSepInput(v);
                          const savedRaw = savedData?.context_sep ?? "";
                          const csvVal =
                            v === "" || v === "\\n\\n"
                              ? savedRaw === "\\n\\n"
                                ? "\\n\\n"
                                : ""
                              : v;
                          onStageRow(row.rowIdx, { context_sep: csvVal });
                        }}
                        onBlur={() => {
                          if (sepInput === "") {
                            setSepInput("\\n\\n");
                          }
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            e.stopPropagation();
                            (e.target as HTMLInputElement).blur();
                          }
                        }}
                        placeholder="\\n\\n"
                        title="Delimiter between prepended context and question stem (use \n for newline)"
                        className="h-6 w-28 bg-white px-2 font-mono text-xs text-slate-900 shadow-none"
                      />
                    </div>
                  )}
                </div>
                <div>{resolvedCtx}</div>
              </div>
            )}

            {/* Helpful hint when standalone is chosen, showing what's sitting above in the same block */}
            {!resolvedCtx &&
              ctxMode === "none" &&
              hasInBlockDb &&
              !row.isFormatSkipped && (
                <div className="p-2.5 rounded-lg bg-slate-50 border border-dashed border-slate-200 text-xs text-slate-500 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <span className="font-semibold text-slate-600">
                      Not prepended (preceding text in same block
                      {topInBlockDbLabel ? ` ${topInBlockDbLabel}` : ""}):{" "}
                    </span>
                    <span className="italic line-clamp-2">
                      {inBlockHintText}
                    </span>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    size="xs"
                    onClick={() => handleSelectContextMode("in_block")}
                    className="bg-white shrink-0 cursor-pointer"
                  >
                    Prepend this
                  </Button>
                </div>
              )}

            {/* Primary Question Text & Override Editor */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                    {isMatrixWithSubItems ? (
                      <>
                        Shared Matrix Stem{" "}
                        <span className="font-normal normal-case text-slate-400">
                          (paired with{" "}
                          {matrixSubItems.length > 1
                            ? `each of the ${matrixSubItems.length} sub-items below`
                            : "the sub-item below"}
                          )
                        </span>
                      </>
                    ) : (
                      `Question Stem${activePairedVariant && !hasStemOverride ? ` (${activePairedVariant.vTag})` : ""}`
                    )}
                  </span>
                  {hasStemOverride && (
                    <Badge
                      variant="outline"
                      className="h-4.5 px-1.5 py-0 text-[10px] font-semibold rounded-sm bg-cyan-50 text-cyan-800 border-cyan-300 uppercase tracking-tight"
                    >
                      Overridden · question_overrides.json
                    </Badge>
                  )}
                </div>

                {onStageOverride && !row.isFormatSkipped && !isCollapsedLoop && (
                  <div className="flex items-center gap-1.5">
                    {!editingStem && !hasStemOverride ? (
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        onClick={() => setEditingStem(true)}
                        className="h-6 bg-white text-slate-700 text-[11px] cursor-pointer"
                      >
                        <Pencil className="w-3 h-3" />
                        Override stem
                      </Button>
                    ) : (
                      <>
                        {hasStemOverride && (
                          <Button
                            type="button"
                            variant="outline"
                            size="xs"
                            onClick={() => setShowOriginalStem((v) => !v)}
                            className="h-6 bg-white text-slate-600 text-[11px] cursor-pointer"
                          >
                            {showOriginalStem
                              ? "Hide original"
                              : "Show original"}
                          </Button>
                        )}
                        {hasStemOverride ? (
                          <Button
                            type="button"
                            variant="outline"
                            size="xs"
                            onClick={() => {
                              stageOverrideUpdate((cur) => {
                                const next = { ...cur };
                                delete next.stem;
                                return next;
                              });
                              setEditingStem(false);
                              setShowOriginalStem(false);
                            }}
                            className="h-6 bg-white text-rose-700 border-rose-200 hover:bg-rose-50 text-[11px] cursor-pointer"
                          >
                            <RotateCcw className="w-3 h-3" />
                            Reset to original
                          </Button>
                        ) : (
                          <Button
                            type="button"
                            variant="ghost"
                            size="xs"
                            onClick={() => setEditingStem(false)}
                            className="h-6 text-slate-500 text-[11px] cursor-pointer"
                          >
                            Close editor
                          </Button>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>

              {editingStem || hasStemOverride ? (
                <div className="space-y-1.5">
                  <Textarea
                    value={effectiveQuestionStem}
                    onChange={(e) => {
                      const val = e.target.value;
                      stageOverrideUpdate((cur) => ({
                        ...cur,
                        stem: val,
                      }));
                    }}
                    rows={Math.min(
                      14,
                      Math.max(3, effectiveQuestionStem.split("\n").length + 1)
                    )}
                    placeholder="Enter overridden question stem (saved to question_overrides.json on Save / Confirm & Next)..."
                    className="w-full bg-cyan-50/30 border-cyan-300 text-sm text-slate-900 leading-relaxed shadow-none"
                  />
                  {detail &&
                    detail.qtext_variants.length > 1 &&
                    !isCollapsedLoop && (
                      <div className="px-2.5 py-1.5 rounded bg-amber-50 border border-amber-200 text-[11px] text-amber-900">
                        <strong>Heads up:</strong> Raw data has{" "}
                        {detail.qtext_variants.length} distinct{" "}
                        <code>QuestionText</code> variants across respondents.
                        Parquet export enforces static 1-variant stems when{" "}
                        <code>override.stem</code> is set.
                      </div>
                    )}
                  {showOriginalStem && (
                    <div className="p-2.5 rounded-lg bg-slate-100/80 border border-slate-200 text-xs text-slate-600 whitespace-pre-wrap leading-relaxed">
                      <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
                        Original Raw QuestionText (untouched in CSV):
                      </div>
                      {rawQuestionStem || "(empty)"}
                    </div>
                  )}
                </div>
              ) : (
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80 text-sm text-slate-900 leading-relaxed whitespace-pre-wrap">
                  {schemaKind === "question_audit"
                    ? effectiveQuestionStem ||
                      d.Context_preview ||
                      "(Empty question stem)"
                    : effectiveQuestionStem || "(Empty question stem)"}
                </div>
              )}
            </div>

            {/* Prominent Matrix Sub-items Inspector (right below the stem, never buried) */}
            {isMatrixWithSubItems && matrixSubItems.length > 0 && (
              <div className="rounded-lg border border-violet-200/90 bg-violet-50/35 p-3 space-y-2.5">
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-bold uppercase tracking-wider text-violet-900">
                      Matrix Sub-items ({matrixSubItems.length})
                    </span>
                    <span className="text-[11px] text-slate-600">
                      {matrixSubItems.length > 1 ? (
                        <>
                          Unrolls into{" "}
                          <strong className="text-slate-800">
                            {matrixSubItems.length} separate questions
                          </strong>{" "}
                          (
                          <code className="font-mono text-[11px] text-violet-900">
                            {matrixSubItems[0].subItemId}
                          </code>
                          {" … "}
                          <code className="font-mono text-[11px] text-violet-900">
                            {matrixSubItems[matrixSubItems.length - 1].subItemId}
                          </code>
                          ) — click any sub-item to inspect its prompt &amp; counts
                        </>
                      ) : (
                        <>Appended directly to the shared stem above</>
                      )}
                    </span>
                  </div>
                  <div className="flex items-center gap-2.5 flex-wrap">
                    {isMatrixLikert &&
                      !isLegacyAutoSubitemSepDataset(dsId) &&
                      onStageOverride && (
                        <div className="flex items-center gap-1 rounded-md border border-violet-200 bg-white px-1.5 py-0.5">
                          <span className="text-[11px] font-medium text-slate-600">
                            Delimiter:
                          </span>
                          {(
                            [
                              { raw: "\\n\\n", label: "\\n\\n", title: "Blank line (default)" },
                              { raw: "\\n", label: "\\n", title: "Single newline" },
                              { raw: " ", label: "␣", title: "Single space" },
                              { raw: ": ", label: ": ", title: "Colon + space" },
                            ] as const
                          ).map((preset) => {
                            const curRaw =
                              effectiveOverride?.subitem_sep ?? defaultSubitemSep(dsId);
                            const active = curRaw === preset.raw;
                            return (
                              <button
                                key={preset.raw}
                                type="button"
                                title={preset.title}
                                onClick={() => {
                                  setSubitemSepInput(preset.raw);
                                  stageOverrideUpdate((cur) => ({
                                    ...cur,
                                    subitem_sep: preset.raw,
                                  }));
                                }}
                                className={[
                                  "px-1.5 py-0.5 rounded font-mono text-[11px] transition-colors cursor-pointer",
                                  active
                                    ? "bg-violet-600 text-white font-semibold"
                                    : "bg-slate-100 text-slate-700 hover:bg-violet-100",
                                ].join(" ")}
                              >
                                {preset.label}
                              </button>
                            );
                          })}
                          <Input
                            type="text"
                            value={subitemSepInput}
                            onChange={(e) => {
                              const v = e.target.value;
                              setSubitemSepInput(v);
                              stageOverrideUpdate((cur) => ({
                                ...cur,
                                subitem_sep: v === "" ? undefined : v,
                              }));
                            }}
                            onBlur={() => {
                              if (subitemSepInput === "") {
                                setSubitemSepInput(defaultSubitemSep(dsId));
                              }
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                e.preventDefault();
                                e.stopPropagation();
                                (e.target as HTMLInputElement).blur();
                              }
                            }}
                            placeholder="\\n\\n"
                            title="Custom delimiter between Matrix stem and sub-item (use \n for newline)"
                            className="h-5 w-16 bg-slate-50 px-1.5 font-mono text-[11px] text-slate-900 shadow-none"
                          />
                        </div>
                      )}
                    {onStageOverride && activeSubItem && (
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        onClick={() => setEditingSubItem((v) => !v)}
                        className="h-6 bg-white text-slate-700 text-[11px] cursor-pointer"
                      >
                        <Pencil className="w-3 h-3" />
                        {editingSubItem || activeSubItem.isSubOverridden
                          ? `Edit ${activeSubItem.subItemId}`
                          : `Override ${activeSubItem.subItemId}`}
                      </Button>
                    )}
                    <label className="inline-flex items-center gap-1.5 text-[11px] font-medium text-slate-700 cursor-pointer select-none">
                      <Checkbox
                        checked={showFullSubItemPrompts}
                        onCheckedChange={(checked) =>
                          setShowFullSubItemPrompts(Boolean(checked))
                        }
                        className="bg-white"
                      />
                      Show full combined prompt per sub-item
                    </label>
                  </div>
                </div>

                {onStageOverride &&
                  activeSubItem &&
                  (editingSubItem || activeSubItem.isSubOverridden) && (
                    <div className="p-2.5 rounded-lg bg-white border border-cyan-300 space-y-1.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[11px] font-semibold text-cyan-900">
                          Override sub-item #{activeSubItem.idx + 1} (
                          {activeSubItem.subItemId})
                          {isMultiRowBipolar
                            ? ' — keep "Left:Right" separator'
                            : ""}
                        </span>
                        {activeSubItem.isSubOverridden && (
                          <Button
                            type="button"
                            variant="outline"
                            size="xs"
                            onClick={() => {
                              stageOverrideUpdate((cur) => {
                                const nextSubs = { ...(cur.subitems ?? {}) };
                                delete nextSubs[String(activeSubItem.idx)];
                                return { ...cur, subitems: nextSubs };
                              });
                              setEditingSubItem(false);
                            }}
                            className="h-5 px-2 text-[11px] text-rose-700 border-rose-200 hover:bg-rose-50 cursor-pointer"
                          >
                            Reset sub-item
                          </Button>
                        )}
                      </div>
                      <Input
                        type="text"
                        value={activeSubItem.rawRowText}
                        onChange={(e) => {
                          const v = e.target.value;
                          stageOverrideUpdate((cur) => ({
                            ...cur,
                            subitems: {
                              ...(cur.subitems ?? {}),
                              [String(activeSubItem.idx)]: v,
                            },
                          }));
                        }}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            e.stopPropagation();
                            (e.target as HTMLInputElement).blur();
                          }
                        }}
                        className="h-7 bg-cyan-50/30 border-cyan-200 text-xs shadow-none"
                      />
                      {activeSubItem.isSubOverridden && (
                        <div className="text-[11px] text-slate-400">
                          Original: <span>{activeSubItem.origRowText}</span>
                        </div>
                      )}
                    </div>
                  )}

                <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1">
                  {matrixSubItems.map((sub) => {
                    const isSelected = sub.idx === safeSubItemIdx;
                    return (
                      <button
                        key={`${sub.idx}-${sub.subItemId}`}
                        type="button"
                        onClick={() => onSelectSubItemIdx(sub.idx)}
                        className={[
                          "w-full text-left p-2.5 rounded-lg border text-xs transition-all flex items-start justify-between gap-3 cursor-pointer",
                          isSelected && matrixSubItems.length > 1
                            ? "bg-white border-violet-400 ring-1 ring-violet-300/70 shadow-2xs"
                            : "bg-white/85 border-slate-200/90 hover:bg-white hover:border-violet-300",
                        ].join(" ")}
                      >
                        <div className="min-w-0 flex-1 space-y-1">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono text-[11px] font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 shrink-0">
                              {sub.idx + 1}. {sub.subItemId}
                            </span>
                            {sub.isSubOverridden && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-cyan-100 text-cyan-800">
                                Overridden
                              </span>
                            )}
                            {isSelected && matrixSubItems.length > 1 && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide bg-violet-100 text-violet-800">
                                Previewing below
                              </span>
                            )}
                            {!showFullSubItemPrompts && (
                              <span className="text-slate-900 font-medium leading-snug">
                                {sub.statementLabel}
                              </span>
                            )}
                          </div>
                          {showFullSubItemPrompts && (
                            <div className="mt-1 p-2 rounded bg-slate-900 text-slate-100 font-mono text-[11px] whitespace-pre-wrap leading-relaxed">
                              {sub.combinedStem}
                              {isMultiRowBipolar && (
                                <div className="mt-1 text-sky-300">
                                  {sub.options.map((o) => `[${o.option}]`).join("  ")}
                                </div>
                              )}
                            </div>
                          )}
                        </div>

                        {sub.totalN > 0 && (
                          <div className="flex items-center gap-2 shrink-0 pt-0.5">
                            <div
                              className="w-20 h-2 rounded-full bg-slate-100 overflow-hidden flex"
                              title={sub.options
                                .map((o) => `${o.option}: ${o.count}`)
                                .join(" | ")}
                            >
                              {sub.options.map((o, oIdx) => {
                                const w = (o.count / sub.totalN) * 100;
                                if (w <= 0) return null;
                                const dropped = dropSet.has(o.rawOption);
                                const shades = [
                                  "bg-violet-300",
                                  "bg-violet-400",
                                  "bg-violet-500",
                                  "bg-slate-400",
                                  "bg-indigo-500",
                                  "bg-indigo-600",
                                  "bg-indigo-700",
                                ];
                                return (
                                  <div
                                    key={`${oIdx}-${o.option}`}
                                    className={
                                      dropped
                                        ? "bg-rose-300"
                                        : shades[oIdx % shades.length]
                                    }
                                    style={{ width: `${w}%` }}
                                  />
                                );
                              })}
                            </div>
                            <span className="font-mono text-[11px] text-slate-500 min-w-[52px] text-right">
                              N={sub.totalN}
                            </span>
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Response Options & Empirical Distribution */}
        {optionItems.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs">
            <div className="flex items-center justify-between gap-2 mb-3 flex-wrap">
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                    Response Options
                    {matrixSubItems.length > 1 && activeSubItem
                      ? ` — ${activeSubItem.subItemId}`
                      : ""}{" "}
                    ({optionItems.length} total
                    {dropSet.size > 0 ? `, ${dropSet.size} dropped` : ""})
                  </h3>
                  {activePairedVariant && (
                    <Badge
                      variant="outline"
                      className="h-4.5 px-1.5 py-0 text-[10px] font-mono font-semibold rounded-sm bg-indigo-50 text-indigo-800 border-indigo-200"
                    >
                      {showPooledOptions
                        ? `pooled across ${resolvedVariants.length} variants`
                        : `${activePairedVariant.vTag} · n=${activePairedVariant.n}`}
                    </Badge>
                  )}
                </div>
                {columns.includes("drop_options") && (
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Click <strong>Drop</strong> on non-substantive options (such as{" "}
                    <em>Don&apos;t know</em>) to omit those responses and re-index the scale
                    {matrixSubItems.length > 1 ? " across all sub-items." : "."}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                {activePairedVariant && (
                  <button
                    type="button"
                    onClick={() => setShowPooledOptions((v) => !v)}
                    className="text-xs text-indigo-700 hover:underline font-medium cursor-pointer"
                  >
                    {showPooledOptions
                      ? `Show ${activePairedVariant.vTag} only`
                      : `Show all ${resolvedVariants.length} variants pooled`}
                  </button>
                )}
                {columns.includes("drop_options") && dropSet.size > 0 && (
                  <button
                    type="button"
                    onClick={clearAllDrops}
                    className="text-xs text-emerald-700 hover:underline font-medium cursor-pointer"
                  >
                    Restore all dropped
                  </button>
                )}
                <span className="text-xs text-slate-400">
                  Total responses: {totalCount.toLocaleString()}
                </span>
              </div>
            </div>
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 text-left">
                  <th className="py-1.5 pr-2 w-8 font-medium">#</th>
                  <th className="py-1.5 pr-3 font-medium">Option label</th>
                  <th className="py-1.5 pr-3 w-20 text-right font-medium">Count</th>
                  <th className="py-1.5 pr-3 w-16 text-right font-medium">Share</th>
                  <th className="py-1.5 pr-3 w-36 font-medium">Distribution</th>
                  {(columns.includes("drop_options") ||
                    (onStageOverride && !isMultiRowBipolar)) && (
                    <th className="py-1.5 w-32 text-right font-medium">Action</th>
                  )}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {optionItems.map((o, i) => {
                  const isDropped = dropSet.has(o.rawOption);
                  const pct = totalCount > 0 ? (o.count / totalCount) * 100 : 0;
                  const isEditingOpt =
                    !isMultiRowBipolar && editingOptionRaw === o.rawOption;
                  return (
                    <tr
                      key={`${i}-${o.rawOption}`}
                      className={isDropped ? "bg-rose-50/40 text-slate-400" : ""}
                    >
                      <td className="py-2 pr-2 font-mono text-slate-400">
                        {i + 1}
                      </td>
                      <td
                        className={`py-2 pr-3 font-medium ${isDropped ? "line-through text-slate-400" : "text-slate-800"}`}
                      >
                        {isEditingOpt ? (
                          <div className="flex items-center gap-1.5">
                            <Input
                              type="text"
                              value={optionDraftText}
                              onChange={(e) => {
                                const val = e.target.value;
                                setOptionDraftText(val);
                                stageOverrideUpdate((cur) => ({
                                  ...cur,
                                  options: {
                                    ...(cur.options ?? {}),
                                    [o.rawOption]: val,
                                  },
                                }));
                              }}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                  e.preventDefault();
                                  e.stopPropagation();
                                  setEditingOptionRaw(null);
                                }
                              }}
                              className="h-6 bg-cyan-50/30 border-cyan-300 text-xs shadow-none max-w-xs"
                            />
                            <Button
                              type="button"
                              variant="outline"
                              size="xs"
                              onClick={() => setEditingOptionRaw(null)}
                              className="h-5 px-2 text-[11px] bg-white cursor-pointer"
                            >
                              Done
                            </Button>
                          </div>
                        ) : (
                          <div className="flex flex-col gap-0.5">
                            <div className="flex items-center gap-1.5 flex-wrap">
                              <span>{o.option}</span>
                              {o.isOverridden && !isMultiRowBipolar && (
                                <span className="px-1.5 py-0.2 rounded bg-cyan-50 border border-cyan-200 text-[10px] font-mono text-cyan-800 no-underline">
                                  raw: {JSON.stringify(o.rawOption)}
                                </span>
                              )}
                            </div>
                            {o.description && (
                              <span className="text-[11px] font-normal text-slate-500">
                                {o.description}
                              </span>
                            )}
                          </div>
                        )}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono text-slate-600">
                        {o.count.toLocaleString()}
                      </td>
                      <td className="py-2 pr-3 text-right font-mono text-slate-500">
                        {totalCount > 0 ? `${pct.toFixed(1)}%` : "—"}
                      </td>
                      <td className="py-2 pr-3">
                        <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              isDropped
                                ? "bg-rose-300"
                                : eff === "score"
                                  ? "bg-violet-500"
                                  : "bg-blue-500"
                            }`}
                            style={{ width: `${Math.min(100, pct)}%` }}
                          />
                        </div>
                      </td>
                      {(columns.includes("drop_options") ||
                        (onStageOverride && !isMultiRowBipolar)) && (
                        <td className="py-2 text-right">
                          <div className="inline-flex items-center justify-end gap-1">
                            {onStageOverride && !isMultiRowBipolar && (
                              <>
                                {o.isOverridden ? (
                                  <Button
                                    type="button"
                                    variant="outline"
                                    size="xs"
                                    onClick={() => {
                                      stageOverrideUpdate((cur) => {
                                        const nextOpts = {
                                          ...(cur.options ?? {}),
                                        };
                                        delete nextOpts[o.rawOption];
                                        return { ...cur, options: nextOpts };
                                      });
                                      if (editingOptionRaw === o.rawOption) {
                                        setEditingOptionRaw(null);
                                      }
                                    }}
                                    className="h-5 px-2 text-[11px] bg-white text-cyan-800 border-cyan-300 hover:bg-cyan-50 cursor-pointer"
                                    title={`Revert option label back to ${JSON.stringify(o.rawOption)}`}
                                  >
                                    Reset
                                  </Button>
                                ) : null}
                                <Button
                                  type="button"
                                  variant="outline"
                                  size="xs"
                                  onClick={() => {
                                    setEditingOptionRaw(o.rawOption);
                                    setOptionDraftText(o.option);
                                  }}
                                  className="h-5 px-2 text-[11px] bg-white text-slate-600 border-slate-200 hover:border-cyan-300 hover:text-cyan-800 cursor-pointer"
                                  title="Override option label in question_overrides.json"
                                >
                                  Rename
                                </Button>
                              </>
                            )}
                            {columns.includes("drop_options") && (
                              <Button
                                type="button"
                                variant="outline"
                                size="xs"
                                onClick={() => toggleDropOption(o.rawOption)}
                                className={[
                                  "h-5 px-2 text-[11px] cursor-pointer",
                                  isDropped
                                    ? "bg-white text-emerald-700 border-emerald-300 hover:bg-emerald-50"
                                    : "bg-white text-slate-600 border-slate-200 hover:border-rose-300 hover:text-rose-700",
                                ].join(" ")}
                              >
                                {isDropped ? "Restore" : "Drop"}
                              </Button>
                            )}
                          </div>
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Live Kev Prompt Preview */}
        {(!row.isFormatSkipped || eff === "score" || eff === "choice") && (
          <div className="bg-slate-900 text-slate-100 rounded-xl p-4 shadow-xs font-mono text-xs leading-relaxed overflow-x-auto">
            <div className="flex items-center justify-between gap-2 flex-wrap mb-2">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 font-sans font-semibold">
                  Live Kev Prompt Preview
                </span>
                {resolvedVariants.length > 1 && activePairedVariant && (
                  <div className="inline-flex items-center gap-1 rounded-md bg-slate-800 border border-slate-700 px-1.5 py-0.5 font-sans text-[11px] text-slate-200">
                    <button
                      type="button"
                      disabled={safeVariantIdx <= 0}
                      onClick={() =>
                        setSelectedVariantIdx(Math.max(0, safeVariantIdx - 1))
                      }
                      className="px-1 text-slate-300 hover:text-white disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                      title="Previous paired variant"
                    >
                      ◀
                    </button>
                    <span className="font-mono text-[11px] text-indigo-300">
                      {activePairedVariant.vTag} ({safeVariantIdx + 1}/
                      {resolvedVariants.length} · n={activePairedVariant.n})
                    </span>
                    <button
                      type="button"
                      disabled={safeVariantIdx >= resolvedVariants.length - 1}
                      onClick={() =>
                        setSelectedVariantIdx(
                          Math.min(
                            resolvedVariants.length - 1,
                            safeVariantIdx + 1
                          )
                        )
                      }
                      className="px-1 text-slate-300 hover:text-white disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                      title="Next paired variant"
                    >
                      ▶
                    </button>
                  </div>
                )}
                {matrixSubItems.length > 1 && activeSubItem && (
                  <div className="inline-flex items-center gap-1 rounded-md bg-slate-800 border border-slate-700 px-1.5 py-0.5 font-sans text-[11px] text-slate-200">
                    <button
                      type="button"
                      disabled={safeSubItemIdx <= 0}
                      onClick={() =>
                        onSelectSubItemIdx(Math.max(0, safeSubItemIdx - 1))
                      }
                      className="px-1 text-slate-300 hover:text-white disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                      title="Previous matrix sub-item"
                    >
                      ◀
                    </button>
                    <span className="font-mono text-[11px] text-violet-300">
                      Sub-item {safeSubItemIdx + 1}/{matrixSubItems.length} (
                      {activeSubItem.subItemId})
                    </span>
                    <button
                      type="button"
                      disabled={safeSubItemIdx >= matrixSubItems.length - 1}
                      onClick={() =>
                        onSelectSubItemIdx(
                          Math.min(matrixSubItems.length - 1, safeSubItemIdx + 1)
                        )
                      }
                      className="px-1 text-slate-300 hover:text-white disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                      title="Next matrix sub-item"
                    >
                      ▶
                    </button>
                  </div>
                )}
              </div>
              <span
                className={[
                  "px-2 py-0.5 rounded text-[10px] font-sans font-semibold uppercase",
                  eff === "score"
                    ? "bg-violet-500/25 text-violet-200"
                    : eff === "choice"
                      ? "bg-blue-500/25 text-blue-200"
                      : "bg-rose-500/25 text-rose-200",
                ].join(" ")}
              >
                {eff === "exclude" ? "excluded from export" : `loss: ${eff}`}
              </span>
            </div>
            <div className="text-emerald-400">&lt;|fim_middle|&gt;</div>
            <div className="whitespace-pre-wrap text-slate-100 my-1">
              {fullStem || "(empty stem)"}
            </div>
            {previewOptionItems.length > 0 ? (
              <div className="space-y-0.5">
                {previewOptionItems.map((o, idx) => {
                  const dropped = dropSet.has(o.rawOption);
                  return (
                    <div
                      key={`${idx}-${o.rawOption}`}
                      className={dropped ? "line-through text-slate-500" : ""}
                    >
                      <span className={dropped ? "text-slate-600" : "text-sky-400"}>
                        &lt;|box_start|&gt;
                      </span>
                      {eff === "choice" && o.description
                        ? `${o.option}: ${o.description}`
                        : o.option}
                      <span className={dropped ? "text-slate-600" : "text-sky-400"}>
                        &lt;|box_end|&gt;
                      </span>
                      {dropped && (
                        <span className="ml-2 text-[10px] text-rose-400 no-underline inline-block">
                          [dropped]
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-slate-500 italic">(no discrete options)</div>
            )}
            <div className="text-emerald-400 mt-1">&lt;|fim_suffix|&gt;</div>
          </div>
        )}

        {/* Non-matrix multi-row fields (e.g. TE/FORM or CS format-skipped items) */}
        {!isMatrixWithSubItems && detail && detail.rows.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2.5">
              Sub-items / Statements ({detail.rows.length})
            </h3>
            <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
              {detail.rows.map((rText, idx) => {
                const bd = detail.row_breakdowns[idx];
                const rowTot = bd
                  ? bd.counts.reduce((a, b) => a + b, 0)
                  : 0;
                return (
                  <div
                    key={`${idx}-${rText}`}
                    className="p-2 rounded-lg bg-slate-50 border border-slate-100 text-xs flex items-center justify-between gap-3"
                  >
                    <span className="text-slate-800">
                      <span className="font-mono text-slate-400 mr-2">
                        {idx + 1}.
                      </span>
                      {rText}
                    </span>
                    {bd && rowTot > 0 && (
                      <span className="font-mono text-[11px] text-slate-500 shrink-0">
                        [{bd.counts.join(", ")}] (N={rowTot})
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Sample Free-Text / Slider Responses (when inspecting format-skipped items) */}
        {detail && detail.sample_texts.length > 0 && (
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-2">
              Sample Respondent Entries
            </h3>
            <ul className="space-y-1 text-xs text-slate-600 font-mono">
              {detail.sample_texts.map((st, idx) => (
                <li
                  key={idx}
                  className="p-2 rounded bg-slate-50 border border-slate-100"
                >
                  {st}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Randomized Stem / DB Variants */}
        {detail &&
          (detail.qtext_variants.length > 1 || detail.db_variants.length > 1) && (
            <details className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs">
              <summary className="text-xs font-bold uppercase tracking-wider text-slate-500 cursor-pointer select-none">
                Randomized Variants ({detail.qtext_variants.length} stem variants,{" "}
                {detail.db_variants.length} DB variants)
              </summary>
              <div className="mt-3 space-y-3 text-xs">
                {detail.qtext_variants.length > 1 && (
                  <div className="space-y-1.5">
                    <div className="font-semibold text-slate-600">
                      QuestionText variants:
                    </div>
                    {detail.qtext_variants.map((v, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded bg-slate-50 border border-slate-100 text-slate-700"
                      >
                        <span className="font-mono text-[11px] text-blue-700 mr-2">
                          [{v.label} · n={v.n}]
                        </span>
                        {v.text}
                      </div>
                    ))}
                  </div>
                )}
                {detail.db_variants.length > 1 && (
                  <div className="space-y-1.5">
                    <div className="font-semibold text-slate-600">
                      In-block DB variants:
                    </div>
                    {detail.db_variants.map((v, idx) => (
                      <div
                        key={idx}
                        className="p-2 rounded bg-slate-50 border border-slate-100 text-slate-700"
                      >
                        <span className="font-mono text-[11px] text-indigo-700 mr-2">
                          [{v.label} · n={v.n}]
                        </span>
                        {v.text}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </details>
          )}

        {/* Optional Reviewer Memo */}
        {columns.includes(noteField) && (
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs space-y-1.5">
            <div className="flex items-baseline justify-between gap-2 flex-wrap">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-500">
                Optional Reviewer Memo
              </label>
              <span className="text-[11px] text-slate-400">
                Saved to CSV when you click Confirm &amp; Next or Save — ignored during Parquet export and training
              </span>
            </div>
            <Input
              type="text"
              value={noteInput}
              onChange={(e) => {
                const v = e.target.value;
                setNoteInput(v);
                onStageRow(row.rowIdx, { [noteField]: v });
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  (e.target as HTMLInputElement).blur();
                }
              }}
              placeholder="Add an optional note about why you kept, changed, or excluded this question..."
              className="h-8 w-full text-xs bg-white shadow-none"
            />
          </div>
        )}
      </div>
    </div>
  );
}
