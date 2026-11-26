"use client";

import React, { useMemo, useState } from "react";
import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Copy,
  FileText,
  Info,
  Loader2,
  Pencil,
  PencilLine,
  RotateCcw,
  Save,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  hasCodebookDiff,
  type SourceCheckItem,
  type SourceCheckPayload,
  type SourceOptionOrigin,
  type SourceOptionRowStatus,
} from "@/lib/sourceCheckTypes";
import {
  defaultSubitemSep,
  getEffectiveType,
  getRawStem,
  normalizeQuestionOverride,
} from "@/lib/triage";
import type {
  AuditRow,
  ItemDetail,
  QuestionOverride,
  SchemaKind,
} from "@/lib/types";
import { CategoryEditor } from "./CategoryEditor";
import { QuestionMarginalsPanel } from "./QuestionMarginalsPanel";
import { TagEditor } from "./TagEditor";

const OPTION_STATUS_STYLE: Record<SourceOptionRowStatus, { label: string; cls: string }> = {
  same: { label: "same", cls: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  overridden: { label: "overridden", cls: "bg-violet-50 text-violet-700 border-violet-200" },
  filled: { label: "placeholder filled", cls: "bg-sky-50 text-sky-700 border-sky-200" },
  dropped: { label: "dropped", cls: "bg-slate-100 text-slate-600 border-slate-300" },
  final_only: { label: "not in codebook", cls: "bg-amber-50 text-amber-800 border-amber-200" },
  source_only: { label: "codebook only", cls: "bg-rose-50 text-rose-700 border-rose-200" },
};

const ORIGIN_TAG: Record<SourceOptionOrigin, { label: string; title: string }> = {
  stata_label: {
    label: "Stata text, as is",
    title: "Exactly the answer text stored in the Stata file",
  },
  stata_label_code: {
    label: "Stata text + scale number",
    title:
      "The Stata file's text with its scale number put in front (e.g. '1. None at all'). The exporter adds the numbers " +
      "because some points on this scale have no text, so every point is numbered.",
  },
  stata_label_filled: {
    label: "Stata text, placeholder filled",
    title:
      "The Stata file's text with a questionnaire placeholder filled in at export time, e.g. " +
      "'I am not [nationality]' becomes 'I am not American' (wvs_export.py US_FILLS)",
  },
  code_only: {
    label: "Scale number only (no text in Stata)",
    title:
      "The Stata file has no words for this point of the scale (it is blank or just the number), so the exporter " +
      "writes the scale number alone, e.g. '2.'",
  },
  override: {
    label: "Renamed in overrides",
    title: "Text changed in question_overrides.json; the Stata file's original text is shown under 'was:'",
  },
};

/** Colours of the PDF highlights (scripts/source_check.py HIGHLIGHT_COLORS). */
const HIGHLIGHT_LEGEND: [string, string][] = [
  ["wording", "rgb(255, 219, 51)"],
  ["same", "rgb(115, 217, 128)"],
  ["overridden", "rgb(184, 148, 255)"],
  ["placeholder filled", "rgb(115, 191, 255)"],
  ["codebook only", "rgb(255, 128, 128)"],
  ["dropped / missing code", "rgb(194, 194, 194)"],
];

export function SourceStatusChips({
  item,
  compact = false,
}: {
  item: SourceCheckItem;
  compact?: boolean;
}) {
  const sizeCls = compact ? "h-4 px-1.5 py-0 text-[10px] rounded-sm" : "text-[10px]";
  return (
    <span className="inline-flex flex-wrap gap-1">
      {item.error !== null && (
        <Badge variant="outline" className={`bg-rose-50 text-rose-700 border-rose-200 ${sizeCls}`}>
          read error
        </Badge>
      )}
      {item.text_status === "changed" && (
        <Badge variant="outline" className={`bg-blue-50 text-blue-700 border-blue-200 ${sizeCls}`}>
          text changed
        </Badge>
      )}
      {item.text_status === "not_in_source" && item.error === null && (
        <Badge variant="outline" className={`bg-slate-100 text-slate-600 border-slate-300 ${sizeCls}`}>
          not in codebook
        </Badge>
      )}
      {item.options_status === "differ" && (
        <Badge variant="outline" className={`bg-amber-50 text-amber-800 border-amber-200 ${sizeCls}`}>
          options differ
        </Badge>
      )}
      {!hasCodebookDiff(item) && (
        <Badge variant="outline" className={`bg-emerald-50 text-emerald-700 border-emerald-200 ${sizeCls}`}>
          identical
        </Badge>
      )}
    </span>
  );
}

function TextDiff({ diff }: { diff: SourceCheckItem["text_diff"] }) {
  return (
    <p className="text-sm leading-relaxed text-slate-800">
      {diff.map(([op, text], i) => (
        <React.Fragment key={i}>
          {i > 0 && " "}
          {op === "eq" && <span>{text}</span>}
          {op === "del" && (
            <span className="bg-rose-100 text-rose-800 line-through decoration-rose-500" title="In the codebook, not in the exported text">
              {text}
            </span>
          )}
          {op === "ins" && (
            <span className="bg-emerald-100 text-emerald-800" title="In the exported text, not in the codebook">
              {text}
            </span>
          )}
        </React.Fragment>
      ))}
    </p>
  );
}

export interface SourcePaneProps {
  dsId: string;
  data: SourceCheckPayload | null;
  loading: boolean;
  error: string | null;
  row: AuditRow | undefined;
  savedData: Record<string, string> | undefined;
  savedOverride?: QuestionOverride;
  effectiveOverride?: QuestionOverride;
  savedTags?: string[];
  allTagCounts?: Array<[string, number]>;
  allCategoryCounts?: Array<[string, number]>;
  detail: ItemDetail | undefined;
  schemaKind: SchemaKind;
  columns: string[];
  draftResetSeq: number;
  positionIndex: number;
  totalFiltered: number;
  hasPrev: boolean;
  hasNext: boolean;
  marginalPersona?: string;
  onSelectMarginalPersona?: (col: string) => void;
  marginalsRefreshSeq?: number;
  onPrev: () => void;
  onNext: () => void;
  onStageRow: (rowIdx: number, patch: Record<string, string>) => void;
  onStageOverride?: (
    rowIdx: number,
    nextOverride: QuestionOverride | null
  ) => void;
  onStageTags?: (rowIdx: number, nextTags: string[]) => void;
  onFilterByTag?: (tag: string) => void;
  onFilterByCategory?: (category: string) => void;
  onSaveRow: (rowIdx: number) => void;
  onDiscardRow: (rowIdx: number) => void;
  onToggleReviewed: (rowIdx: number, nextReviewed: boolean) => void;
  onConfirmAndNext: () => void;
  onEditInQuestions: () => void;
}

interface SourceMiddlePaneProps {
  dsId: string;
  data: SourceCheckPayload;
  item: SourceCheckItem;
  loading: boolean;
  row: AuditRow;
  savedData: Record<string, string> | undefined;
  savedOverride?: QuestionOverride;
  effectiveOverride?: QuestionOverride;
  savedTags: string[];
  allTagCounts: Array<[string, number]>;
  allCategoryCounts: Array<[string, number]>;
  detail: ItemDetail | undefined;
  schemaKind: SchemaKind;
  columns: string[];
  positionIndex: number;
  totalFiltered: number;
  hasPrev: boolean;
  hasNext: boolean;
  subTab: "codebook" | "marginals";
  onChangeSubTab: (tab: "codebook" | "marginals") => void;
  marginalPersona: string;
  onSelectMarginalPersona?: (col: string) => void;
  marginalsRefreshSeq: number;
  onPrev: () => void;
  onNext: () => void;
  onStageRow: (rowIdx: number, patch: Record<string, string>) => void;
  onStageOverride?: (
    rowIdx: number,
    nextOverride: QuestionOverride | null
  ) => void;
  onStageTags?: (rowIdx: number, nextTags: string[]) => void;
  onFilterByTag?: (tag: string) => void;
  onFilterByCategory?: (category: string) => void;
  onSaveRow: (rowIdx: number) => void;
  onDiscardRow: (rowIdx: number) => void;
  onToggleReviewed: (rowIdx: number, nextReviewed: boolean) => void;
  onConfirmAndNext: () => void;
  onEditInQuestions: () => void;
}

function SourceMiddlePane({
  dsId,
  data,
  item,
  loading,
  row,
  savedData,
  savedOverride,
  effectiveOverride,
  savedTags,
  allTagCounts,
  allCategoryCounts,
  detail,
  schemaKind,
  columns,
  positionIndex,
  totalFiltered,
  hasPrev,
  hasNext,
  subTab,
  onChangeSubTab,
  marginalPersona,
  onSelectMarginalPersona,
  marginalsRefreshSeq,
  onPrev,
  onNext,
  onStageRow,
  onStageOverride,
  onStageTags,
  onFilterByTag,
  onFilterByCategory,
  onSaveRow,
  onDiscardRow,
  onToggleReviewed,
  onConfirmAndNext,
  onEditInQuestions,
}: SourceMiddlePaneProps) {
  const d = row.data;
  const eff = getEffectiveType(schemaKind, d);
  const noteField = schemaKind === "question_audit" ? "review_note" : "note";
  const isDirty = Boolean(row.isDirty);
  const dirtyFields = row.dirtyFields ?? [];

  const [editingStem, setEditingStem] = useState(() =>
    Boolean(effectiveOverride && effectiveOverride.stem !== undefined)
  );
  const [showOriginalStem, setShowOriginalStem] = useState(false);
  const [editingOptionRaw, setEditingOptionRaw] = useState<string | null>(null);
  const [optionDraftText, setOptionDraftText] = useState("");
  const [noteInput, setNoteInput] = useState(() =>
    schemaKind === "question_audit" ? d.review_note || "" : d.note || ""
  );

  const dropSet = useMemo(
    () =>
      new Set(
        (d.drop_options || "")
          .split("|")
          .map((s) => s.trim())
          .filter(Boolean)
      ),
    [d.drop_options]
  );

  const rawQuestionStem = getRawStem(schemaKind, d);
  const hasStemOverride = effectiveOverride?.stem !== undefined;
  const effectiveQuestionStem = hasStemOverride
    ? (effectiveOverride?.stem ?? "")
    : rawQuestionStem;
  const fullStem = effectiveQuestionStem.trim();

  const stageOverrideUpdate = (
    updater: (cur: QuestionOverride) => QuestionOverride
  ) => {
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

  const handleSetDecision = (target: "score" | "choice" | "exclude") => {
    if (schemaKind === "question_audit") {
      if (target === "exclude") {
        onStageRow(row.rowIdx, { include: "False", kev_type: "" });
      } else {
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
    onStageRow(row.rowIdx, { drop_options: Array.from(next).join(" | ") });
  };

  const clearAllDrops = () => {
    onStageRow(row.rowIdx, { drop_options: "" });
  };

  const empiricalOptions = useMemo(() => {
    const rawList =
      detail && detail.options.length > 0
        ? detail.options
        : (d.Options_preview || d.options || "")
            .replace(/^\[\d+\]\s*/, "")
            .split(" | ")
            .filter(Boolean)
            .map((o) => ({ option: o.trim(), count: 0, description: null }));
    return rawList.map((o) => ({
      rawOption: o.option,
      option: effectiveOverride?.options?.[o.option] ?? o.option,
      description: o.description ?? null,
      isOverridden: Boolean(
        effectiveOverride?.options && o.option in effectiveOverride.options
      ),
      count: o.count,
    }));
  }, [detail, d.Options_preview, d.options, effectiveOverride?.options]);

  const countByRawOption = useMemo(() => {
    const map = new Map<string, number>();
    for (const o of detail?.options ?? []) {
      map.set(o.option, o.count);
    }
    return map;
  }, [detail]);

  const totalEmpiricalCount = useMemo(
    () => (detail?.options ?? []).reduce((acc, o) => acc + o.count, 0),
    [detail]
  );

  const optionRowRawSet = useMemo(() => {
    const set = new Set<string>();
    for (const r of item.option_rows) {
      if (r.status === "source_only") continue;
      const rawOpt = r.original ?? r.final;
      if (rawOpt !== null) set.add(rawOpt);
    }
    return set;
  }, [item.option_rows]);

  const unlistedEmpiricalOptions = useMemo(
    () => empiricalOptions.filter((o) => !optionRowRawSet.has(o.rawOption)),
    [empiricalOptions, optionRowRawSet]
  );

  const hasEmpiricalCounts = Boolean(detail && detail.options.length > 0);
  const canEditOptions =
    columns.includes("drop_options") || Boolean(onStageOverride);
  const missingCodes = item.source_options.filter((o) => o.missing_code);
  const exportProblemText = row.exportProblem || item.export_problem || "";

  return (
    <div className="flex-1 min-w-0 flex flex-col h-full overflow-hidden bg-slate-50">
      {/* Sticky Top Action Bar */}
      <div className="bg-white border-b border-slate-200 px-3 py-1.5 flex items-center gap-1.5 shrink-0 flex-wrap">
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="outline"
            size="icon-xs"
            onClick={onPrev}
            disabled={!hasPrev}
            className="bg-white cursor-pointer"
            title="Previous (k / ↑)"
          >
            <ChevronLeft className="w-3.5 h-3.5 text-slate-700" />
          </Button>
          <span className="text-[11px] font-mono text-slate-500 min-w-[48px] text-center">
            {positionIndex >= 0
              ? `${positionIndex + 1}/${totalFiltered}`
              : `—/${totalFiltered}`}
          </span>
          <Button
            type="button"
            variant="outline"
            size="icon-xs"
            onClick={onNext}
            disabled={!hasNext}
            className="bg-white cursor-pointer"
            title="Next (j / ↓)"
          >
            <ChevronRight className="w-3.5 h-3.5 text-slate-700" />
          </Button>
        </div>

        <div className="h-4 w-px bg-slate-200" />

        {/* Modeling Type Segmented Buttons */}
        <div
          className="inline-flex rounded-md border border-slate-200 p-0.5 bg-slate-100/80"
          role="group"
          aria-label="Question type"
        >
          <button
            type="button"
            onClick={() => handleSetDecision("score")}
            title="Stage as Score (key: s)"
            className={[
              "px-2 py-0.5 text-[11px] font-semibold rounded transition-all cursor-pointer",
              eff === "score"
                ? "bg-violet-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            Score <kbd className="opacity-75 font-mono text-[9px]">s</kbd>
          </button>
          <button
            type="button"
            onClick={() => handleSetDecision("choice")}
            title="Stage as Choice (key: c)"
            className={[
              "px-2 py-0.5 text-[11px] font-semibold rounded transition-all cursor-pointer",
              eff === "choice"
                ? "bg-blue-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            Choice <kbd className="opacity-75 font-mono text-[9px]">c</kbd>
          </button>
          <button
            type="button"
            onClick={() => handleSetDecision("exclude")}
            title="Stage as Exclude (key: x)"
            className={[
              "px-2 py-0.5 text-[11px] font-semibold rounded transition-all cursor-pointer",
              eff === "exclude"
                ? row.isFormatSkipped
                  ? "bg-slate-500 text-white shadow-xs"
                  : "bg-rose-600 text-white shadow-xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            Exclude <kbd className="opacity-75 font-mono text-[9px]">x</kbd>
          </button>
        </div>

        {columns.includes("reversed") && (
          <label className="inline-flex items-center gap-1 text-[11px] text-slate-700 cursor-pointer select-none">
            <Checkbox
              checked={d.reversed === "True"}
              onCheckedChange={(val) =>
                onStageRow(row.rowIdx, {
                  reversed: val ? "True" : "False",
                })
              }
              className="bg-white"
            />
            Reversed
          </label>
        )}

        {loading && (
          <span className="inline-flex items-center gap-1 text-[11px] text-slate-500">
            <Loader2 className="w-3 h-3 animate-spin" />
            Refreshing...
          </span>
        )}

        {/* Unsaved Draft Controls */}
        {isDirty && (
          <>
            <Badge
              variant="outline"
              className="rounded-md text-[10px] font-bold bg-orange-100 text-orange-900 border-orange-300 uppercase tracking-tight"
            >
              Unsaved ({dirtyFields.length})
            </Badge>
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() => onDiscardRow(row.rowIdx)}
              className="bg-white text-slate-600 hover:bg-slate-50 text-[11px] cursor-pointer"
              title="Discard unsaved changes on this question (Esc)"
            >
              <RotateCcw className="w-3 h-3" />
              Discard
            </Button>
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() => onSaveRow(row.rowIdx)}
              className="border-emerald-300 bg-emerald-50 text-emerald-900 hover:bg-emerald-100 text-[11px] cursor-pointer"
              title="Save changes to CSV/overrides and refresh codebook diff (Ctrl/Cmd+S)"
            >
              <Save className="w-3 h-3 text-emerald-700" />
              Save
            </Button>
          </>
        )}

        <div className="ml-auto flex items-center gap-1.5 flex-wrap">
          {!isDirty && (
            <Button
              type="button"
              variant="outline"
              size="xs"
              onClick={() => onToggleReviewed(row.rowIdx, !row.reviewed)}
              className={[
                "text-[11px] cursor-pointer",
                row.reviewed
                  ? "bg-emerald-50 text-emerald-800 border-emerald-200 hover:bg-emerald-100"
                  : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50",
              ].join(" ")}
              title="Toggle reviewed without modifying CSV (Space)"
            >
              <CheckCircle2
                className={`w-3 h-3 ${row.reviewed ? "text-emerald-600" : "text-slate-400"}`}
              />
              {row.reviewed ? "Reviewed" : "Mark reviewed"}
            </Button>
          )}

          <Button
            type="button"
            size="xs"
            onClick={onConfirmAndNext}
            className={[
              "text-[11px] font-semibold text-white cursor-pointer",
              isDirty
                ? "bg-emerald-600 hover:bg-emerald-700"
                : "bg-slate-900 hover:bg-slate-800",
            ].join(" ")}
            title={
              isDirty
                ? "Save staged changes, mark reviewed, and advance (Enter)"
                : "Mark reviewed and advance (Enter)"
            }
          >
            <Check className="w-3 h-3" />
            {isDirty ? "Save & Next" : "Confirm & Next"}
            <kbd
              className={[
                "ml-0.5 px-1 rounded text-[9px] font-mono",
                isDirty ? "bg-emerald-800" : "bg-slate-700",
              ].join(" ")}
            >
              ↵
            </kbd>
          </Button>
        </div>
      </div>

      {/* Sub-view Tab Bar: Codebook & Audit vs Persona Marginals */}
      <div className="bg-white border-b border-slate-200 px-3 flex items-center justify-between gap-2 shrink-0">
        <div role="tablist" aria-label="Source detail view" className="flex items-center gap-1 -mb-px">
          <button
            type="button"
            role="tab"
            aria-selected={subTab === "codebook"}
            onClick={() => onChangeSubTab("codebook")}
            className={[
              "inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold border-b-2 transition-colors cursor-pointer",
              subTab === "codebook"
                ? "border-emerald-700 text-emerald-800"
                : "border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300",
            ].join(" ")}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Codebook &amp; Audit
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={subTab === "marginals"}
            onClick={() => onChangeSubTab("marginals")}
            className={[
              "inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold border-b-2 transition-colors cursor-pointer",
              subTab === "marginals"
                ? "border-blue-600 text-blue-700"
                : "border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300",
            ].join(" ")}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Persona Marginals
          </button>
        </div>

        <Button
          type="button"
          size="xs"
          variant="ghost"
          onClick={onEditInQuestions}
          title="Switch to the full-width Questions tab (key: e)"
          className="h-6 px-2 text-slate-500 hover:text-slate-900 text-[11px] cursor-pointer shrink-0"
        >
          <PencilLine className="w-3 h-3" />
          Questions <kbd className="ml-0.5 px-1 rounded bg-slate-100 border border-slate-200 font-mono text-[9px]">e</kbd>
        </Button>
      </div>

      {/* Scrollable Middle Content */}
      <section className="flex-1 min-w-0 overflow-y-auto p-4 space-y-3.5">
        {subTab === "marginals" ? (
          <QuestionMarginalsPanel
            dsId={dsId}
            questionId={
              (schemaKind === "question_audit" ? d.QuestionID : d.var) ||
              d.QuestionID ||
              d.var ||
              item.v
            }
            study={d.study || undefined}
            questionStem={fullStem || item.final_text}
            selectedPersona={marginalPersona}
            onSelectPersona={(col) => onSelectMarginalPersona?.(col)}
            refreshSeq={marginalsRefreshSeq}
            notFieldedIn={row.notFieldedIn}
          />
        ) : (
          <>
            {item.error !== null && (
              <div className="rounded-md border border-rose-200 bg-rose-50 p-3 text-xs text-rose-800">
                {item.error}
              </div>
            )}

            {/* Metadata, Draft Diff, Tags & Banners Card */}
            <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-2xs space-y-2.5">
              {isDirty && savedData && (
                <div className="flex items-center justify-between gap-2 flex-wrap px-2.5 py-2 rounded-lg bg-orange-50 border border-orange-200 text-orange-950 text-xs">
                  <div className="flex items-center gap-1.5 flex-wrap min-w-0">
                    <span className="font-bold text-orange-900 uppercase text-[10px] tracking-tight">
                      Unsaved draft:
                    </span>
                    {dirtyFields.map((col) => {
                      if (col === "tags") {
                        return (
                          <span
                            key={col}
                            className="px-1.5 py-0.5 rounded bg-white border border-orange-200 font-mono text-[11px] text-slate-800 truncate max-w-xs"
                          >
                            <strong>tags</strong>:{" "}
                            <span className="text-slate-400">
                              [{savedTags.map((t) => `#${t}`).join(", ")}]
                            </span>{" "}
                            →{" "}
                            <span className="text-orange-800 font-semibold">
                              [{(row.tags ?? []).map((t) => `#${t}`).join(", ")}]
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
                            className="px-1.5 py-0.5 rounded bg-white border border-orange-200 font-mono text-[11px] text-slate-800 truncate max-w-xs"
                          >
                            <strong>{col}</strong>:{" "}
                            <span className="text-orange-800 font-semibold">
                              {label}
                            </span>
                          </span>
                        );
                      }
                      return (
                        <span
                          key={col}
                          className="px-1.5 py-0.5 rounded bg-white border border-orange-200 font-mono text-[11px] text-slate-800 truncate max-w-xs"
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
                    Codebook diff &amp; PDF refresh on <strong>Save</strong> (⌘S)
                  </span>
                </div>
              )}

              <div className="flex items-start justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <h2 className="text-base font-bold font-mono text-slate-900">
                    {item.v}
                  </h2>
                  {(item.is_persona || detail?.is_persona) && (
                    <Badge
                      variant="outline"
                      className="bg-purple-50 text-purple-800 border-purple-200 text-[10px] font-semibold"
                    >
                      PERSONA
                    </Badge>
                  )}
                  <SourceStatusChips item={item} />
                  {schemaKind === "question_audit" && d.family && (
                    <span className="px-1.5 py-0.5 text-[11px] font-mono rounded bg-slate-100 text-slate-700">
                      {d.family}
                    </span>
                  )}
                  {d.BlockName && (
                    <span className="text-xs text-slate-500">
                      Block: <strong className="text-slate-700">{d.BlockName}</strong>
                    </span>
                  )}
                  {d.n_answered && (
                    <span className="text-xs text-slate-500 font-mono">
                      N={d.n_answered}
                    </span>
                  )}
                  {d.data_label && (
                    <span className="text-xs text-slate-500">{d.data_label}</span>
                  )}
                </div>
                <span className="text-[11px] font-mono text-slate-400 shrink-0">
                  row #{row.rowIdx}
                </span>
              </div>

              {columns.includes("category") && (
                <CategoryEditor
                  category={d.category ?? ""}
                  allCategoryCounts={allCategoryCounts}
                  onChange={(nextCategory) =>
                    onStageRow(row.rowIdx, { category: nextCategory })
                  }
                  onFilterByCategory={onFilterByCategory}
                />
              )}

              {onStageTags && (
                <TagEditor
                  tags={row.tags ?? []}
                  allTagCounts={allTagCounts}
                  onChange={(nextTags) => onStageTags(row.rowIdx, nextTags)}
                  onFilterByTag={onFilterByTag}
                />
              )}

              {/* Format-skipped banner */}
              {row.isFormatSkipped && eff !== "score" && eff !== "choice" && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-xs">
                  <Info className="w-4 h-4 text-slate-400 shrink-0" />
                  <span>
                    <strong>Auto-skipped ({row.formatSkipLabel}):</strong>{" "}
                    {row.formatSkipReason}.
                  </span>
                </div>
              )}

              {/* Not fielded in a country */}
              {row.notFieldedIn.length > 0 && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-100 border border-slate-200 text-slate-700 text-xs">
                  <Info className="w-4 h-4 text-slate-400 shrink-0" />
                  <span>
                    <strong>Not fielded in {row.notFieldedIn.join(", ")}:</strong> no valid
                    answers in that country&apos;s sample, so it is left out of that
                    country&apos;s parquet.
                  </span>
                </div>
              )}

              {/* Export-blocking banner */}
              {exportProblemText && (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-red-50 border border-red-300 text-red-900 text-xs">
                  <AlertTriangle className="w-4 h-4 text-red-600 shrink-0" />
                  <span>
                    <strong>Blocks export:</strong> {exportProblemText}. Set this question to
                    Exclude, otherwise the Parquet rebuild fails.
                  </span>
                </div>
              )}

              {/* Suspect reasons banner */}
              {row.isSuspect && (
                <div className="flex items-center gap-1.5 flex-wrap px-3 py-2 rounded-lg bg-amber-50/90 border border-amber-200 text-amber-900 text-xs">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                  <span className="font-semibold">Suspect:</span>
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

              {/* Codebook stem review warnings */}
              {item.warnings.length > 0 && (
                <div className="rounded-lg border border-orange-300 bg-orange-50 p-2.5 text-xs text-orange-900 space-y-1">
                  <div className="font-semibold">Flagged for stem review</div>
                  <ul className="list-disc pl-4 space-y-0.5">
                    {item.warnings.map((w, i) => (
                      <li key={i}>{w}</li>
                    ))}
                  </ul>
                  <div className="text-[11px] text-orange-800/90">
                    Override the stem below or set to <strong>Exclude</strong> (key: <code>x</code>).
                  </div>
                </div>
              )}
            </div>

            {/* Question Wording Comparison + Inline Stem Override */}
            <div className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-2xs space-y-3">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                    Question wording
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
                {onStageOverride && !row.isFormatSkipped && (
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {item.source_text !== null &&
                      effectiveQuestionStem.trim() !== item.source_text.trim() && (
                        <Button
                          type="button"
                          variant="outline"
                          size="xs"
                          onClick={() => {
                            const srcText = item.source_text ?? "";
                            stageOverrideUpdate((cur) => ({
                              ...cur,
                              stem: srcText,
                            }));
                            setEditingStem(true);
                          }}
                          title="Copy the codebook wording above into question_overrides.json"
                          className="h-6 bg-white text-emerald-800 border-emerald-200 hover:bg-emerald-50 text-[11px] cursor-pointer"
                        >
                          <Copy className="w-3 h-3" />
                          Use codebook wording
                        </Button>
                      )}
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
                            {showOriginalStem ? "Hide original" : "Show original"}
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

              {item.source_text === null ? (
                <div className="text-xs text-slate-500">
                  No block for this variable in the codebook.
                </div>
              ) : (
                <div>
                  <div className="text-[11px] text-slate-500 mb-1">
                    Codebook{item.page !== null ? ` (PDF p. ${item.page})` : ""}
                  </div>
                  <p className="text-sm text-slate-700 leading-relaxed">
                    {item.source_text}
                  </p>
                </div>
              )}

              <div className="space-y-1.5">
                <div className="text-[11px] text-slate-500">
                  Exported{hasStemOverride ? " (override draft/saved)" : ""}
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
                        12,
                        Math.max(3, effectiveQuestionStem.split("\n").length + 1)
                      )}
                      placeholder="Enter overridden question stem (saved to question_overrides.json on Save)..."
                      className="w-full bg-cyan-50/30 border-cyan-300 text-sm text-slate-900 leading-relaxed shadow-none"
                    />
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
                  <p className="text-sm text-slate-800 leading-relaxed whitespace-pre-wrap">
                    {effectiveQuestionStem || item.final_text || "(Empty question stem)"}
                  </p>
                )}
              </div>

              {item.source_text !== null && item.text_status === "changed" && (
                <div>
                  <div className="text-[11px] text-slate-500 mb-1">
                    Differences
                    {dirtyFields.includes("override.stem") ? " (last saved)" : ""}{" "}
                    (<span className="bg-rose-100 text-rose-800 line-through">codebook only</span>{" "}
                    <span className="bg-emerald-100 text-emerald-800">exported only</span>)
                  </div>
                  <TextDiff diff={item.text_diff} />
                </div>
              )}
            </div>

            {/* Answer Options: Codebook Comparison + Counts/% + Rename/Drop */}
            <div className="rounded-xl border border-slate-200 bg-white p-3.5 shadow-2xs space-y-2.5">
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                    Answer options
                    {empiricalOptions.length > 0
                      ? ` (${empiricalOptions.length} in data${dropSet.size > 0 ? `, ${dropSet.size} dropped` : ""})`
                      : ""}
                    {item.options_status === "n/a" ? " · no codebook comparison" : ""}
                  </span>
                </div>
                {columns.includes("drop_options") && dropSet.size > 0 && (
                  <Button
                    type="button"
                    variant="outline"
                    size="xs"
                    onClick={clearAllDrops}
                    className="h-6 bg-white text-rose-700 border-rose-200 hover:bg-rose-50 text-[11px] cursor-pointer"
                  >
                    Clear all drops ({dropSet.size})
                  </Button>
                )}
              </div>

              {item.option_rows.length > 0 && (
                <div className="divide-y divide-slate-100 border-t border-slate-100">
                  <div className="grid grid-cols-2 gap-2.5 py-1 text-[11px] font-medium text-slate-500">
                    <div>Codebook</div>
                    <div className="truncate" title={data.final_source ?? undefined}>
                      Exported
                      {data.final_source ? (
                        <span className="font-normal text-slate-400"> ({data.final_source})</span>
                      ) : null}
                    </div>
                  </div>
                  {item.option_rows.map((r, i) => {
                    const rawOption =
                      r.status === "source_only" ? null : (r.original ?? r.final);
                    const liveDropped =
                      rawOption !== null && dropSet.has(rawOption);
                    const liveOverridden =
                      rawOption !== null &&
                      Boolean(
                        effectiveOverride?.options &&
                          rawOption in effectiveOverride.options
                      );
                    const liveText =
                      rawOption !== null
                        ? (effectiveOverride?.options?.[rawOption] ?? rawOption)
                        : null;
                    const empCount =
                      rawOption !== null
                        ? countByRawOption.get(rawOption)
                        : undefined;
                    const empPct =
                      empCount !== undefined && totalEmpiricalCount > 0
                        ? (empCount / totalEmpiricalCount) * 100
                        : null;
                    const savedDropped = r.status === "dropped";
                    const draftDropChanged =
                      rawOption !== null && liveDropped !== savedDropped;
                    const draftRenameChanged =
                      rawOption !== null &&
                      !liveDropped &&
                      liveText !== (r.final ?? rawOption);

                    return (
                      <div
                        key={i}
                        className={[
                          "py-2 space-y-1.5 text-xs",
                          liveDropped ? "bg-rose-50/35 -mx-1 px-1 rounded" : "",
                        ].join(" ")}
                      >
                        <div className="grid grid-cols-2 gap-2.5 items-start">
                          <div className="text-slate-700 break-words">
                            {r.source ?? <span className="text-slate-300">—</span>}
                          </div>
                          <div className="text-slate-800 break-words min-w-0">
                            {rawOption === null ? (
                              <span className="text-slate-300">—</span>
                            ) : editingOptionRaw === rawOption ? (
                              <div className="flex items-center gap-1">
                                <Input
                                  type="text"
                                  value={optionDraftText}
                                  autoFocus
                                  onChange={(e) => {
                                    const val = e.target.value;
                                    setOptionDraftText(val);
                                    stageOverrideUpdate((cur) => ({
                                      ...cur,
                                      options: {
                                        ...(cur.options ?? {}),
                                        [rawOption]: val,
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
                                  className="h-6 bg-cyan-50/30 border-cyan-300 text-xs shadow-none min-w-0 flex-1"
                                />
                                <Button
                                  type="button"
                                  variant="outline"
                                  size="xs"
                                  onClick={() => setEditingOptionRaw(null)}
                                  className="h-5 px-1.5 text-[10px] bg-white cursor-pointer shrink-0"
                                >
                                  Done
                                </Button>
                              </div>
                            ) : (
                              <>
                                <span
                                  className={
                                    liveDropped
                                      ? "line-through text-slate-400"
                                      : "text-slate-900 font-medium"
                                  }
                                >
                                  {liveText}
                                </span>
                                {liveOverridden && (
                                  <div className="text-[10px] text-cyan-800">
                                    raw: {rawOption}
                                  </div>
                                )}
                                {!liveOverridden && r.original !== null && (
                                  <div className="text-[10px] text-slate-500">
                                    {liveDropped ? "dropped: " : "was: "}
                                    {r.original}
                                  </div>
                                )}
                              </>
                            )}
                          </div>
                        </div>

                        {/* Compact metadata + distribution + actions strip (never overflows horizontally) */}
                        <div className="flex items-center justify-between gap-2 flex-wrap">
                          <div className="flex items-center gap-1.5 flex-wrap">
                            <Badge
                              variant="outline"
                              className={`h-4 px-1.5 py-0 text-[10px] rounded-sm whitespace-nowrap ${OPTION_STATUS_STYLE[r.status].cls}`}
                            >
                              {OPTION_STATUS_STYLE[r.status].label}
                            </Badge>
                            {draftDropChanged && (
                              <span className="px-1 py-0 rounded bg-orange-100 text-orange-900 border border-orange-200 text-[9px] font-semibold whitespace-nowrap">
                                {liveDropped ? "draft: drop" : "draft: restore"}
                              </span>
                            )}
                            {draftRenameChanged && (
                              <span className="px-1 py-0 rounded bg-cyan-50 text-cyan-800 border border-cyan-200 text-[9px] font-semibold whitespace-nowrap">
                                draft: renamed
                              </span>
                            )}
                            {!liveDropped && r.origin !== null && !draftRenameChanged && (
                              <span
                                title={ORIGIN_TAG[r.origin].title}
                                className="inline-block rounded border border-slate-200 bg-slate-50 px-1 text-[9px] text-slate-500"
                              >
                                {ORIGIN_TAG[r.origin].label}
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-2 ml-auto">
                            {hasEmpiricalCounts && empCount !== undefined && (
                              <div className="flex items-center gap-1.5 font-mono text-[10px] text-slate-600">
                                {empPct !== null && (
                                  <div className="w-12 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                                    <div
                                      className={`h-full rounded-full ${
                                        liveDropped
                                          ? "bg-rose-300"
                                          : eff === "score"
                                            ? "bg-violet-500"
                                            : "bg-blue-500"
                                      }`}
                                      style={{ width: `${Math.min(100, empPct)}%` }}
                                    />
                                  </div>
                                )}
                                <span>
                                  {empCount.toLocaleString()}
                                  {empPct !== null && (
                                    <span className="text-slate-400 ml-0.5">
                                      ({empPct.toFixed(1)}%)
                                    </span>
                                  )}
                                </span>
                              </div>
                            )}

                            {canEditOptions && rawOption !== null && (
                              <div className="inline-flex items-center gap-1">
                                {onStageOverride && liveOverridden && (
                                  <Button
                                    type="button"
                                    variant="outline"
                                    size="xs"
                                    onClick={() => {
                                      stageOverrideUpdate((cur) => {
                                        const nextOpts = {
                                          ...(cur.options ?? {}),
                                        };
                                        delete nextOpts[rawOption];
                                        return { ...cur, options: nextOpts };
                                      });
                                      if (editingOptionRaw === rawOption) {
                                        setEditingOptionRaw(null);
                                      }
                                    }}
                                    className="h-5 px-1.5 text-[10px] bg-white text-cyan-800 border-cyan-300 hover:bg-cyan-50 cursor-pointer"
                                    title={`Revert option label back to ${JSON.stringify(rawOption)}`}
                                  >
                                    Reset
                                  </Button>
                                )}
                                {onStageOverride && (
                                  <Button
                                    type="button"
                                    variant="outline"
                                    size="xs"
                                    onClick={() => {
                                      setEditingOptionRaw(rawOption);
                                      setOptionDraftText(liveText ?? rawOption);
                                    }}
                                    className="h-5 px-1.5 text-[10px] bg-white text-slate-600 border-slate-200 hover:border-cyan-300 hover:text-cyan-800 cursor-pointer"
                                    title="Override option label in question_overrides.json"
                                  >
                                    Rename
                                  </Button>
                                )}
                                {columns.includes("drop_options") && (
                                  <Button
                                    type="button"
                                    variant="outline"
                                    size="xs"
                                    onClick={() => toggleDropOption(rawOption)}
                                    className={[
                                      "h-5 px-1.5 text-[10px] cursor-pointer",
                                      liveDropped
                                        ? "bg-white text-emerald-700 border-emerald-300 hover:bg-emerald-50"
                                        : "bg-white text-slate-600 border-slate-200 hover:border-rose-300 hover:text-rose-700",
                                    ].join(" ")}
                                  >
                                    {liveDropped ? "Restore" : "Drop"}
                                  </Button>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Empirical options from survey data not listed in codebook comparison (e.g. currently excluded rows) */}
              {unlistedEmpiricalOptions.length > 0 && (
                <div className="pt-2 border-t border-slate-100 space-y-1">
                  <div className="text-[11px] font-medium text-slate-500">
                    Survey options in data
                    {!item.included ? " (currently saved as excluded)" : ""}
                  </div>
                  <div className="divide-y divide-slate-100">
                    {unlistedEmpiricalOptions.map((o) => {
                      const isDropped = dropSet.has(o.rawOption);
                      const pct =
                        totalEmpiricalCount > 0
                          ? (o.count / totalEmpiricalCount) * 100
                          : 0;
                      return (
                        <div
                          key={o.rawOption}
                          className={[
                            "py-1.5 space-y-1 text-xs",
                            isDropped ? "bg-rose-50/35 -mx-1 px-1 rounded" : "",
                          ].join(" ")}
                        >
                          {editingOptionRaw === o.rawOption ? (
                            <div className="flex items-center gap-1.5">
                              <Input
                                type="text"
                                value={optionDraftText}
                                autoFocus
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
                                className="h-6 bg-cyan-50/30 border-cyan-300 text-xs shadow-none flex-1"
                              />
                              <Button
                                type="button"
                                variant="outline"
                                size="xs"
                                onClick={() => setEditingOptionRaw(null)}
                                className="h-5 px-2 text-[10px] bg-white cursor-pointer"
                              >
                                Done
                              </Button>
                            </div>
                          ) : (
                            <div className="flex items-center justify-between gap-2 flex-wrap">
                              <div className="min-w-0">
                                <span
                                  className={
                                    isDropped
                                      ? "line-through text-slate-400"
                                      : "text-slate-800 font-medium"
                                  }
                                >
                                  {o.option}
                                </span>
                                {o.isOverridden && (
                                  <div className="text-[10px] text-cyan-800">
                                    raw: {o.rawOption}
                                  </div>
                                )}
                              </div>
                              <div className="flex items-center gap-2 ml-auto">
                                <div className="flex items-center gap-1.5 font-mono text-[10px] text-slate-600">
                                  {totalEmpiricalCount > 0 && (
                                    <div className="w-12 h-1.5 rounded-full bg-slate-100 overflow-hidden">
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
                                  )}
                                  <span>
                                    {o.count.toLocaleString()}
                                    {totalEmpiricalCount > 0 && (
                                      <span className="text-slate-400 ml-0.5">
                                        ({pct.toFixed(1)}%)
                                      </span>
                                    )}
                                  </span>
                                </div>
                                {canEditOptions && (
                                  <div className="inline-flex items-center gap-1">
                                    {onStageOverride && o.isOverridden && (
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
                                        className="h-5 px-1.5 text-[10px] bg-white text-cyan-800 border-cyan-300 hover:bg-cyan-50 cursor-pointer"
                                      >
                                        Reset
                                      </Button>
                                    )}
                                    {onStageOverride && (
                                      <Button
                                        type="button"
                                        variant="outline"
                                        size="xs"
                                        onClick={() => {
                                          setEditingOptionRaw(o.rawOption);
                                          setOptionDraftText(o.option);
                                        }}
                                        className="h-5 px-1.5 text-[10px] bg-white text-slate-600 border-slate-200 hover:border-cyan-300 hover:text-cyan-800 cursor-pointer"
                                      >
                                        Rename
                                      </Button>
                                    )}
                                    {columns.includes("drop_options") && (
                                      <Button
                                        type="button"
                                        variant="outline"
                                        size="xs"
                                        onClick={() => toggleDropOption(o.rawOption)}
                                        className={[
                                          "h-5 px-1.5 text-[10px] cursor-pointer",
                                          isDropped
                                            ? "bg-white text-emerald-700 border-emerald-300 hover:bg-emerald-50"
                                            : "bg-white text-slate-600 border-slate-200 hover:border-rose-300 hover:text-rose-700",
                                        ].join(" ")}
                                      >
                                        {isDropped ? "Restore" : "Drop"}
                                      </Button>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {missingCodes.length > 0 && (
                <div className="text-[11px] text-slate-500">
                  Missing codes in the codebook (not answer options):{" "}
                  {missingCodes.map((o) => o.text).join(" · ")}
                </div>
              )}
            </div>

            {/* Optional Reviewer Memo */}
            {columns.includes(noteField) && (
              <div className="bg-white rounded-xl border border-slate-200 p-3.5 shadow-2xs space-y-1.5">
                <div className="flex items-baseline justify-between gap-2 flex-wrap">
                  <label className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                    Optional Reviewer Memo
                  </label>
                  <span className="text-[11px] text-slate-400">
                    Saved to CSV on Save or Save &amp; Next — ignored during Parquet export
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

            {/* Live Kev Prompt Preview */}
            {(!row.isFormatSkipped || eff === "score" || eff === "choice") && (
              <div className="bg-slate-900 text-slate-100 rounded-xl p-3.5 shadow-xs font-mono text-xs leading-relaxed overflow-x-auto">
                <div className="flex items-center justify-between gap-2 flex-wrap mb-2">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 font-sans font-semibold">
                    Live Kev Prompt Preview
                  </span>
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
                {empiricalOptions.length > 0 ? (
                  <div className="space-y-0.5">
                    {empiricalOptions.map((o, idx) => {
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
          </>
        )}
      </section>
    </div>
  );
}

export function SourcePane({
  dsId,
  data,
  loading,
  error,
  row,
  savedData,
  savedOverride,
  effectiveOverride,
  savedTags = [],
  allTagCounts = [],
  allCategoryCounts = [],
  detail,
  schemaKind,
  columns,
  draftResetSeq,
  positionIndex,
  totalFiltered,
  hasPrev,
  hasNext,
  marginalPersona = "",
  onSelectMarginalPersona,
  marginalsRefreshSeq = 0,
  onPrev,
  onNext,
  onStageRow,
  onStageOverride,
  onStageTags,
  onFilterByTag,
  onFilterByCategory,
  onSaveRow,
  onDiscardRow,
  onToggleReviewed,
  onConfirmAndNext,
  onEditInQuestions,
}: SourcePaneProps) {
  const [subTab, setSubTab] = useState<"codebook" | "marginals">("codebook");
  const selectedKey = row?.rowKey ?? null;
  const item = data && selectedKey !== null ? data.items[selectedKey] : undefined;

  if (error) {
    return (
      <div className="flex-1 min-w-0 p-6">
        <div className="rounded-md border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 whitespace-pre-wrap">
          Could not load the codebook comparison for {dsId}
          {"\n\n"}
          {error}
        </div>
      </div>
    );
  }
  if (loading && !data) {
    return (
      <div className="flex-1 min-w-0 flex items-center justify-center gap-2 text-sm text-slate-500">
        <Loader2 className="w-4 h-4 animate-spin" /> Comparing questions with the codebook...
      </div>
    );
  }
  if (!data || !data.configured) {
    return (
      <div className="flex-1 min-w-0 p-6 text-sm text-slate-600">
        No codebook comparison is configured for <strong>{dsId}</strong>. Providers are registered in{" "}
        <code>scripts/source_check.py</code> (PROVIDERS).
      </div>
    );
  }

  if (!row || !item || selectedKey === null) {
    return (
      <div className="flex-1 min-w-0 flex items-center justify-center p-6 text-sm text-slate-500">
        {totalFiltered === 0
          ? "No questions match the current filters."
          : "Select a question from the sidebar to compare with the codebook."}
      </div>
    );
  }

  const doc = data.documents[item.doc];
  const pdfSrc =
    doc && item.page !== null
      ? `/api/source-pdf?ds=${encodeURIComponent(dsId)}&doc=${encodeURIComponent(item.doc)}&row=${encodeURIComponent(selectedKey)}&rev=${encodeURIComponent(data.revision)}#page=${item.page}`
      : null;

  return (
    <div className="flex-1 min-w-0 flex overflow-hidden">
      {/* Middle: unified codebook comparison + full audit controls */}
      <SourceMiddlePane
        key={`${row.rowKey}:${draftResetSeq}`}
        dsId={dsId}
        data={data}
        item={item}
        loading={loading}
        row={row}
        savedData={savedData}
        savedOverride={savedOverride}
        effectiveOverride={effectiveOverride}
        savedTags={savedTags}
        allTagCounts={allTagCounts}
        allCategoryCounts={allCategoryCounts}
        detail={detail}
        schemaKind={schemaKind}
        columns={columns}
        positionIndex={positionIndex}
        totalFiltered={totalFiltered}
        hasPrev={hasPrev}
        hasNext={hasNext}
        subTab={subTab}
        onChangeSubTab={setSubTab}
        marginalPersona={marginalPersona}
        onSelectMarginalPersona={onSelectMarginalPersona}
        marginalsRefreshSeq={marginalsRefreshSeq}
        onPrev={onPrev}
        onNext={onNext}
        onStageRow={onStageRow}
        onStageOverride={onStageOverride}
        onStageTags={onStageTags}
        onFilterByTag={onFilterByTag}
        onFilterByCategory={onFilterByCategory}
        onSaveRow={onSaveRow}
        onDiscardRow={onDiscardRow}
        onToggleReviewed={onToggleReviewed}
        onConfirmAndNext={onConfirmAndNext}
        onEditInQuestions={onEditInQuestions}
      />

      {/* Right: codebook PDF at the question's page */}
      <section className="w-[44%] shrink-0 border-l border-slate-200 bg-slate-100 flex flex-col">
        <div className="px-3 py-1.5 border-b border-slate-200 bg-white text-[11px] text-slate-600 flex items-center gap-1.5 truncate">
          <FileText className="w-3.5 h-3.5 shrink-0" />
          <span className="truncate">{doc ? doc.label : "No codebook page"}</span>
          {item.page !== null && <span className="shrink-0">· PDF p. {item.page}</span>}
        </div>
        {pdfSrc && (
          <div className="px-3 py-1 border-b border-slate-200 bg-white flex flex-wrap gap-x-2 gap-y-0.5 text-[10px] text-slate-600">
            {HIGHLIGHT_LEGEND.map(([label, color]) => (
              <span key={label} className="inline-flex items-center gap-1">
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: color }} />
                {label}
              </span>
            ))}
          </div>
        )}
        {pdfSrc ? (
          <iframe key={pdfSrc} src={pdfSrc} title="Codebook" className="flex-1 w-full bg-white" />
        ) : (
          <div className="flex-1 flex items-center justify-center text-xs text-slate-500">No codebook page for this row.</div>
        )}
      </section>
    </div>
  );
}
