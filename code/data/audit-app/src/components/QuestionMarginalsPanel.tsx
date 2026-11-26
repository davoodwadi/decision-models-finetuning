"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  BarChart3,
  Columns3,
  Layers,
  Loader2,
  Table2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  MarginalGroupRow,
  QuestionMarginalsPayload,
} from "@/lib/types";

interface QuestionMarginalsPanelProps {
  dsId: string;
  questionId: string;
  subitemId?: string;
  study?: string;
  questionStem?: string;
  selectedPersona: string;
  onSelectPersona: (col: string) => void;
  refreshSeq?: number;
  /** Countries with zero valid answers (ISSP): explains why the question is absent from the parquet. */
  notFieldedIn?: string[];
}

const QUICK_PERSONA_COLS = new Set([
  "political_party",
  "US_PRTY",
  "political_ideology",
  "PARTY_LR",
  "gender",
  "SEX",
  "age",
  "AGE",
  "education",
  "EDULEVEL",
  "race",
  "RACE",
  "region",
  "US_REG",
  "religion",
  "RELIGGRP",
  "income",
  "US_INC",
]);

// Diverging palette for ordinal score items (interpolated by index).
const SCORE_PALETTE_5 = ["#2563eb", "#60a5fa", "#cbd5e1", "#fb923c", "#dc2626"];
const SCORE_PALETTE_7 = [
  "#1d4ed8",
  "#3b82f6",
  "#93c5fd",
  "#cbd5e1",
  "#fdba74",
  "#f97316",
  "#dc2626",
];

// Qualitative palette for choice options and persona groups.
const QUAL_PALETTE = [
  "#2563eb",
  "#e11d48",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#ea580c",
  "#4f46e5",
  "#16a34a",
  "#be185d",
  "#0d9488",
  "#ca8a04",
  "#9333ea",
  "#0284c7",
  "#65a30d",
  "#dc2626",
];

function optionColor(idx: number, total: number, isScore: boolean): string {
  if (!isScore || total <= 2) {
    return QUAL_PALETTE[idx % QUAL_PALETTE.length];
  }
  if (total === 5) return SCORE_PALETTE_5[idx];
  if (total === 7) return SCORE_PALETTE_7[idx];
  // Sample smoothly across SCORE_PALETTE_7
  const pos = total === 1 ? 0 : (idx / (total - 1)) * (SCORE_PALETTE_7.length - 1);
  return SCORE_PALETTE_7[Math.round(pos)];
}

function groupColor(idx: number, isMissing: boolean): string {
  if (isMissing) return "#94a3b8";
  return QUAL_PALETTE[idx % QUAL_PALETTE.length];
}

function truncateLabel(text: string, maxLen = 42): string {
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1)}…`;
}

function formatDeltaPp(val: number): string {
  const sign = val > 0 ? "+" : "";
  return `${sign}${val.toFixed(1)} pp`;
}

export function QuestionMarginalsPanel({
  dsId,
  questionId,
  subitemId,
  study,
  questionStem,
  selectedPersona,
  onSelectPersona,
  refreshSeq = 0,
  notFieldedIn = [],
}: QuestionMarginalsPanelProps) {
  const [data, setData] = useState<QuestionMarginalsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedArm, setSelectedArm] = useState<string>("");
  const [overrideV, setOverrideV] = useState<string>("");
  const [chartMode, setChartMode] = useState<"stacked" | "grouped">("stacked");
  const [includeMissing, setIncludeMissing] = useState(false);
  const [hideSmallGroups, setHideSmallGroups] = useState(false);
  const [showTable, setShowTable] = useState(true);
  const [hoveredOptIdx, setHoveredOptIdx] = useState<number | null>(null);

  // Reset question-scoped filters when switching question or sub-item
  useEffect(() => {
    setSelectedArm("");
    setOverrideV("");
    setHoveredOptIdx(null);
  }, [dsId, questionId, subitemId, study]);

  const effectiveSubitem = overrideV || subitemId || "";

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    const params = new URLSearchParams({
      ds: dsId,
      questionId,
    });
    if (effectiveSubitem) params.set("subitemId", effectiveSubitem);
    if (study) params.set("study", study);
    if (selectedPersona) params.set("persona", selectedPersona);
    if (selectedArm) params.set("arm", selectedArm);

    fetch(`/api/personas/marginals?${params.toString()}`, {
      cache: "no-store",
    })
      .then(async (res) => {
        const body = await res.json();
        if (!res.ok) {
          throw new Error(body.error || `HTTP ${res.status}`);
        }
        return body as QuestionMarginalsPayload;
      })
      .then((payload) => {
        if (cancelled) return;
        setData(payload);
        setLoading(false);
        // Auto-select grouped bars when there are many choice options (> 8) so bars stay legible
        if (
          payload.found &&
          payload.options &&
          payload.options.length > 8 &&
          payload.questionType !== "score"
        ) {
          setChartMode("grouped");
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    dsId,
    questionId,
    effectiveSubitem,
    study,
    selectedPersona,
    selectedArm,
    refreshSeq,
  ]);

  const activePersonaCol = data?.selectedPersona?.col ?? selectedPersona;

  const quickPersonas = useMemo(() => {
    const avail = data?.availablePersonas ?? [];
    const quick = avail.filter((p) => QUICK_PERSONA_COLS.has(p.col));
    return quick.length > 0 ? quick.slice(0, 8) : avail.slice(0, 6);
  }, [data?.availablePersonas]);

  const isMarginal = Boolean(data?.isMarginal);

  const visibleGroups: MarginalGroupRow[] = useMemo(() => {
    const groups = data?.groups ?? [];
    return groups.filter((g) => {
      if (g.isMissing && !includeMissing) return false;
      if (!isMarginal && hideSmallGroups && !g.isMissing && g.n < 15) return false;
      return true;
    });
  }, [data?.groups, includeMissing, hideSmallGroups, isMarginal]);

  const missingGroup = useMemo(
    () => (data?.groups ?? []).find((g) => g.isMissing) ?? null,
    [data?.groups]
  );

  const smallGroupCount = useMemo(
    () =>
      isMarginal
        ? 0
        : (data?.groups ?? []).filter((g) => !g.isMissing && g.n < 15).length,
    [data?.groups, isMarginal]
  );

  const isScore = data?.questionType === "score";
  const options = data?.options ?? [];

  // Shared max percentage for Grouped Bar view so option panels share a consistent scale
  const groupedMaxPct = useMemo(() => {
    let maxP = 0;
    for (const opt of options) {
      if (opt.overallPct > maxP) maxP = opt.overallPct;
    }
    for (const g of visibleGroups) {
      for (const s of g.shares) {
        if (s > maxP) maxP = s;
      }
    }
    return Math.max(10, Math.ceil(maxP / 10) * 10);
  }, [options, visibleGroups]);

  return (
    <div className="space-y-4">
      {/* Compact Question Context Header */}
      {questionStem && (
        <div className="bg-white rounded-lg border border-slate-200 px-4 py-2.5">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 mb-0.5">
            Question
          </div>
          <p className="text-xs text-slate-800 leading-relaxed whitespace-pre-wrap line-clamp-3">
            {questionStem}
          </p>
        </div>
      )}

      {/* Persona & Display Controls */}
      <div className="bg-white rounded-lg border border-slate-200 p-3.5 space-y-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-700 whitespace-nowrap">
              Persona attribute
            </span>
            <Select
              value={activePersonaCol || "__none__"}
              disabled={loading || !data || data.availablePersonas.length === 0}
              onValueChange={(val) => {
                if (val && val !== "__none__") onSelectPersona(val);
              }}
            >
              <SelectTrigger
                size="sm"
                className="bg-white text-xs min-w-[230px] shadow-none"
              >
                <SelectValue>
                  {(val: string | null) => {
                    const hit = data?.availablePersonas.find(
                      (p) => p.col === val
                    );
                    if (!hit) return "Select persona…";
                    return `${hit.promptLabel} (${hit.nLevels})`;
                  }}
                </SelectValue>
              </SelectTrigger>
              <SelectContent align="start">
                {(data?.availablePersonas ?? []).map((p) => (
                  <SelectItem key={p.col} value={p.col} className="text-xs">
                    <span className="font-medium">{p.promptLabel}</span>
                    <span className="ml-2 font-mono text-[10px] text-slate-400">
                      {p.col} · {p.nLevels} levels
                    </span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Sub-item selector if multiple parquet v's match */}
          {data && data.matchedVs.length > 1 && (
            <div className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <Select
                value={data.activeV ?? data.matchedVs[0]}
                onValueChange={(val) => {
                  if (val) setOverrideV(val);
                }}
              >
                <SelectTrigger
                  size="sm"
                  className="bg-white font-mono text-xs shadow-none"
                >
                  <SelectValue>
                    {(val: string | null) => val ?? data.activeV}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent align="start">
                  {data.matchedVs.map((v) => (
                    <SelectItem key={v} value={v} className="font-mono text-xs">
                      {v}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Arm selector for multi-arm experiments */}
          {data && data.availableArms.length > 1 && (
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-slate-500">Arm</span>
              <Select
                value={selectedArm || "__all__"}
                onValueChange={(val) =>
                  setSelectedArm(!val || val === "__all__" ? "" : val)
                }
              >
                <SelectTrigger size="sm" className="bg-white text-xs shadow-none">
                  <SelectValue>
                    {(val: string | null) =>
                      !val || val === "__all__"
                        ? `All arms (${data.availableArms.length})`
                        : val
                    }
                  </SelectValue>
                </SelectTrigger>
                <SelectContent align="start">
                  <SelectItem value="__all__" className="text-xs">
                    All arms pooled ({data.availableArms.length})
                  </SelectItem>
                  {data.availableArms.map((a) => (
                    <SelectItem key={a} value={a} className="text-xs">
                      {a}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Chart mode toggle */}
          <div className="ml-auto inline-flex rounded-lg border border-slate-200 p-0.5 bg-slate-100/80">
            <button
              type="button"
              onClick={() => setChartMode("stacked")}
              className={[
                "inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded-md transition-colors cursor-pointer",
                chartMode === "stacked"
                  ? "bg-white text-slate-900 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900",
              ].join(" ")}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Stacked 100%
            </button>
            <button
              type="button"
              onClick={() => setChartMode("grouped")}
              className={[
                "inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold rounded-md transition-colors cursor-pointer",
                chartMode === "grouped"
                  ? "bg-white text-slate-900 shadow-2xs"
                  : "text-slate-600 hover:text-slate-900",
              ].join(" ")}
            >
              <Columns3 className="w-3.5 h-3.5" />
              By option
            </button>
          </div>
        </div>

        {/* Quick-pick persona pills */}
        {quickPersonas.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-slate-400 mr-0.5">Quick pick:</span>
            {quickPersonas.map((p) => {
              const active = p.col === activePersonaCol;
              return (
                <button
                  key={p.col}
                  type="button"
                  onClick={() => onSelectPersona(p.col)}
                  className={[
                    "px-2 py-0.5 rounded-full text-[11px] font-medium border transition-colors cursor-pointer",
                    active
                      ? "bg-blue-600 text-white border-blue-600"
                      : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100",
                  ].join(" ")}
                >
                  {p.promptLabel}
                </button>
              );
            })}
          </div>
        )}

        {/* Filter checkboxes */}
        {data?.found && (
          <div className="flex flex-wrap items-center gap-4 pt-1 border-t border-slate-100 text-xs text-slate-600">
            {missingGroup && (
              <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
                <Checkbox
                  checked={includeMissing}
                  onCheckedChange={(c) => setIncludeMissing(Boolean(c))}
                  className="bg-white"
                />
                <span>
                  Show Missing (NA){" "}
                  <span className="font-mono text-[11px] text-slate-400">
                    (n={missingGroup.n})
                  </span>
                </span>
              </label>
            )}
            {smallGroupCount > 0 && (
              <label className="inline-flex items-center gap-1.5 cursor-pointer select-none">
                <Checkbox
                  checked={hideSmallGroups}
                  onCheckedChange={(c) => setHideSmallGroups(Boolean(c))}
                  className="bg-white"
                />
                <span>
                  Hide small groups (n &lt; 15){" "}
                  <span className="font-mono text-[11px] text-slate-400">
                    ({smallGroupCount})
                  </span>
                </span>
              </label>
            )}
            <label className="inline-flex items-center gap-1.5 cursor-pointer select-none ml-auto">
              <Checkbox
                checked={showTable}
                onCheckedChange={(c) => setShowTable(Boolean(c))}
                className="bg-white"
              />
              <Table2 className="w-3.5 h-3.5 text-slate-400" />
              <span>{isMarginal ? "Show % table" : "Show % (n) table"}</span>
            </label>
          </div>
        )}
      </div>

      {/* Loading / Error / Empty states */}
      {loading && (
        <div className="bg-white rounded-lg border border-slate-200 p-10 flex items-center justify-center gap-2 text-xs text-slate-500">
          <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
          Computing marginal distributions from parquet…
        </div>
      )}

      {!loading && error && (
        <div className="bg-rose-50 rounded-lg border border-rose-200 p-4 flex items-start gap-2.5 text-xs text-rose-800">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
          <div>
            <div className="font-semibold">Could not load marginals</div>
            <div className="mt-0.5 font-mono text-[11px]">{error}</div>
          </div>
        </div>
      )}

      {!loading && !error && data && !data.found && (
        <div className="bg-amber-50 rounded-lg border border-amber-200 p-4 flex items-start gap-2.5 text-xs text-amber-900">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
          <div>
            <div className="font-semibold">Not available in current parquet</div>
            <div className="mt-0.5">
              {notFieldedIn.length > 0
                ? `Not fielded in ${notFieldedIn.join(", ")}: no valid answers in that sample, so the question is left out of ${data.parquetFile}.`
                : data.reason}
            </div>
          </div>
        </div>
      )}

      {/* Main Results */}
      {!loading && !error && data && data.found && (
        <>
          {/* Separation Summary Strip */}
          {data.separation && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5">
              <div className="bg-white rounded-lg border border-slate-200 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                  {isMarginal
                    ? "Max TVD (aggregate)"
                    : `Max TVD (n ≥ ${data.separation.minGroupN})`}
                </div>
                <div className="text-lg font-bold text-slate-900 font-mono mt-0.5">
                  {data.separation.maxTvdPct.toFixed(1)} pp
                </div>
                <div className="text-[11px] text-slate-500 truncate mt-0.5">
                  {data.separation.tvdPair
                    ? `${data.separation.tvdPair[0]} vs. ${data.separation.tvdPair[1]}`
                    : "Need ≥ 2 groups"}
                </div>
              </div>

              <div className="bg-white rounded-lg border border-slate-200 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                  Largest option shift
                </div>
                {data.separation.maxOptionDiff ? (
                  <>
                    <div className="text-lg font-bold text-blue-700 font-mono mt-0.5">
                      +{data.separation.maxOptionDiff.deltaPct.toFixed(1)} pp
                    </div>
                    <div
                      className="text-[11px] text-slate-500 truncate mt-0.5"
                      title={`${data.separation.maxOptionDiff.option}: ${data.separation.maxOptionDiff.highGroup} (${data.separation.maxOptionDiff.highPct}%) vs ${data.separation.maxOptionDiff.lowGroup} (${data.separation.maxOptionDiff.lowPct}%)`}
                    >
                      {truncateLabel(data.separation.maxOptionDiff.option, 30)}
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-slate-400 mt-1">—</div>
                )}
              </div>

              {isScore && data.separation.maxMeanDiff ? (
                <div className="bg-white rounded-lg border border-slate-200 p-3">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                    Max mean gap (1–{options.length})
                  </div>
                  <div className="text-lg font-bold text-violet-700 font-mono mt-0.5">
                    Δ {data.separation.maxMeanDiff.delta.toFixed(2)}
                  </div>
                  <div
                    className="text-[11px] text-slate-500 truncate mt-0.5"
                    title={`${data.separation.maxMeanDiff.highGroup} (${data.separation.maxMeanDiff.highMean}) vs ${data.separation.maxMeanDiff.lowGroup} (${data.separation.maxMeanDiff.lowMean})`}
                  >
                    {data.separation.maxMeanDiff.highGroup} ({data.separation.maxMeanDiff.highMean.toFixed(2)}) vs.{" "}
                    {data.separation.maxMeanDiff.lowGroup} ({data.separation.maxMeanDiff.lowMean.toFixed(2)})
                  </div>
                </div>
              ) : (
                <div className="bg-white rounded-lg border border-slate-200 p-3">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                    Question type
                  </div>
                  <div className="mt-1 flex items-center gap-1.5">
                    <Badge
                      variant="outline"
                      className={
                        isScore
                          ? "bg-violet-50 text-violet-800 border-violet-200 text-xs"
                          : "bg-blue-50 text-blue-800 border-blue-200 text-xs"
                      }
                    >
                      {data.questionType}
                    </Badge>
                    <span className="text-xs text-slate-500">
                      {options.length} options
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono truncate mt-1">
                    v={data.activeV}
                  </div>
                </div>
              )}

              <div className="bg-white rounded-lg border border-slate-200 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                  {isMarginal ? "Subgroups (aggregate)" : "Respondents"}
                </div>
                <div className="text-lg font-bold text-slate-900 font-mono mt-0.5">
                  {(data.overallN ?? 0).toLocaleString()}
                </div>
                <div className="text-[11px] text-slate-500 truncate mt-0.5 font-mono">
                  {data.parquetFile}
                </div>
              </div>
            </div>
          )}

          {/* Visualization Card */}
          <div className="bg-white rounded-lg border border-slate-200 p-4 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-xs font-bold text-slate-900">
                  Marginal response distribution by{" "}
                  <span className="text-blue-700">
                    {data.selectedPersona?.promptLabel}
                  </span>
                </h3>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  {chartMode === "stacked"
                    ? "Hover any legend item or bar segment to inspect option share and shift vs. overall."
                    : "Each panel shows the share selecting that option across persona groups (dashed line = overall share)."}
                </p>
              </div>
              {isScore && data.overallMean != null && (
                <span className="text-[11px] font-mono text-slate-500 bg-slate-50 border border-slate-200 rounded px-2 py-0.5">
                  Overall mean: {data.overallMean.toFixed(2)} / {options.length}
                </span>
              )}
            </div>

            {chartMode === "stacked" ? (
              <div className="space-y-3">
                {/* Option Legend */}
                <div className="flex flex-wrap gap-x-3 gap-y-1.5 pb-2 border-b border-slate-100">
                  {options.map((opt, idx) => {
                    const color = optionColor(idx, options.length, isScore);
                    const dimmed =
                      hoveredOptIdx !== null && hoveredOptIdx !== idx;
                    return (
                      <button
                        key={opt.code}
                        type="button"
                        onMouseEnter={() => setHoveredOptIdx(idx)}
                        onMouseLeave={() => setHoveredOptIdx(null)}
                        className={[
                          "inline-flex items-center gap-1.5 text-[11px] transition-opacity cursor-default",
                          dimmed ? "opacity-35" : "opacity-100",
                        ].join(" ")}
                      >
                        <span
                          className="w-2.5 h-2.5 rounded-xs shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <span className="text-slate-700">
                          {truncateLabel(opt.text, 38)}
                        </span>
                        <span className="font-mono text-[10px] text-slate-400">
                          ({opt.overallPct.toFixed(1)}%)
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Overall Reference Row */}
                <div className="space-y-1.5">
                  <div className="grid grid-cols-[190px_1fr_auto] items-center gap-3">
                    <div className="flex items-baseline justify-between gap-2 text-xs">
                      <span className="font-bold text-slate-800">
                        {isMarginal ? "Group mean" : "Overall"}
                      </span>
                      <span className="font-mono text-[10px] text-slate-400">
                        {isMarginal
                          ? `${data.overallN ?? 0} grp`
                          : `n=${(data.overallN ?? 0).toLocaleString()}`}
                      </span>
                    </div>
                    <div className="h-6 w-full rounded overflow-hidden flex bg-slate-100 ring-1 ring-slate-200">
                      {options.map((opt, oi) => {
                        if (opt.overallPct <= 0) return null;
                        const color = optionColor(oi, options.length, isScore);
                        const dimmed =
                          hoveredOptIdx !== null && hoveredOptIdx !== oi;
                        return (
                          <div
                            key={opt.code}
                            onMouseEnter={() => setHoveredOptIdx(oi)}
                            onMouseLeave={() => setHoveredOptIdx(null)}
                            style={{
                              width: `${opt.overallPct}%`,
                              backgroundColor: color,
                            }}
                            title={
                              isMarginal
                                ? `Group mean · ${opt.text}: ${opt.overallPct.toFixed(1)}%`
                                : `Overall · ${opt.text}: ${opt.overallPct.toFixed(1)}% (${opt.overallCount}/${data.overallN})`
                            }
                            className={[
                              "h-full flex items-center justify-center text-[10px] font-mono font-semibold text-white transition-opacity overflow-hidden px-0.5",
                              dimmed ? "opacity-25" : "opacity-100",
                            ].join(" ")}
                          >
                            {opt.overallPct >= 8
                              ? `${Math.round(opt.overallPct)}%`
                              : ""}
                          </div>
                        );
                      })}
                    </div>
                    <div className="w-[72px] text-right font-mono text-[11px] font-semibold text-slate-700">
                      {isScore && data.overallMean != null
                        ? `μ ${data.overallMean.toFixed(2)}`
                        : ""}
                    </div>
                  </div>

                  <div className="border-b border-dashed border-slate-200 my-1" />

                  {/* Persona Group Rows */}
                  {visibleGroups.map((g) => (
                    <div
                      key={g.level}
                      className={[
                        "grid grid-cols-[190px_1fr_auto] items-center gap-3",
                        g.isMissing ? "opacity-75" : "",
                      ].join(" ")}
                    >
                      <div
                        className="flex items-baseline justify-between gap-2 text-xs min-w-0"
                        title={isMarginal ? g.level : `${g.level} (n=${g.n})`}
                      >
                        <span
                          className={[
                            "truncate",
                            g.isMissing
                              ? "italic text-slate-500"
                              : "font-medium text-slate-800",
                          ].join(" ")}
                        >
                          {g.level}
                        </span>
                        <span className="font-mono text-[10px] text-slate-400 shrink-0">
                          {isMarginal ? "agg" : `n=${g.n.toLocaleString()}`}
                        </span>
                      </div>

                      <div className="h-6 w-full rounded overflow-hidden flex bg-slate-100">
                        {options.map((opt, oi) => {
                          const share = g.shares[oi] ?? 0;
                          if (share <= 0) return null;
                          const color = optionColor(oi, options.length, isScore);
                          const dimmed =
                            hoveredOptIdx !== null && hoveredOptIdx !== oi;
                          const diffPp = share - opt.overallPct;
                          return (
                            <div
                              key={opt.code}
                              onMouseEnter={() => setHoveredOptIdx(oi)}
                              onMouseLeave={() => setHoveredOptIdx(null)}
                              style={{
                                width: `${share}%`,
                                backgroundColor: color,
                              }}
                              title={
                                isMarginal
                                  ? `${g.level} · ${opt.text}: ${share.toFixed(1)}% (${formatDeltaPp(diffPp)} vs mean)`
                                  : `${g.level} · ${opt.text}: ${share.toFixed(1)}% (${g.counts[oi]}/${g.n}, ${formatDeltaPp(diffPp)} vs overall)`
                              }
                              className={[
                                "h-full flex items-center justify-center text-[10px] font-mono font-semibold text-white transition-opacity overflow-hidden px-0.5",
                                dimmed ? "opacity-25" : "opacity-100",
                              ].join(" ")}
                            >
                              {share >= 8 ? `${Math.round(share)}%` : ""}
                            </div>
                          );
                        })}
                      </div>

                      <div className="w-[72px] text-right font-mono text-[11px] text-slate-600">
                        {isScore && g.meanScore != null ? (
                          <span
                            title={
                              data.overallMean != null
                                ? `Δ vs overall: ${(g.meanScore - data.overallMean >= 0 ? "+" : "")}${(g.meanScore - data.overallMean).toFixed(2)}`
                                : undefined
                            }
                          >
                            μ {g.meanScore.toFixed(2)}
                          </span>
                        ) : (
                          ""
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              /* Grouped Comparison View (by option) */
              <div className="space-y-3">
                {/* Group Color Legend */}
                <div className="flex flex-wrap gap-x-3 gap-y-1.5 pb-2 border-b border-slate-100">
                  {visibleGroups.map((g, gi) => (
                    <span
                      key={g.level}
                      className="inline-flex items-center gap-1.5 text-[11px] text-slate-700"
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-xs shrink-0"
                        style={{
                          backgroundColor: groupColor(gi, g.isMissing),
                        }}
                      />
                      <span className={g.isMissing ? "italic text-slate-500" : ""}>
                        {g.level}
                      </span>
                      {!isMarginal && (
                        <span className="font-mono text-[10px] text-slate-400">
                          (n={g.n})
                        </span>
                      )}
                    </span>
                  ))}
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
                  {options.map((opt, oi) => (
                    <div
                      key={opt.code}
                      className="rounded-md border border-slate-200/80 bg-slate-50/40 p-3 space-y-2"
                    >
                      <div className="flex items-baseline justify-between gap-2">
                        <span
                          className="text-xs font-semibold text-slate-800 leading-snug"
                          title={opt.text}
                        >
                          {opt.text}
                        </span>
                        <span className="font-mono text-[11px] text-slate-500 shrink-0">
                          Overall {opt.overallPct.toFixed(1)}%
                        </span>
                      </div>

                      <div className="space-y-1.5">
                        {visibleGroups.map((g, gi) => {
                          const share = g.shares[oi] ?? 0;
                          const diffPp = share - opt.overallPct;
                          const barW = Math.min(
                            100,
                            (share / groupedMaxPct) * 100
                          );
                          const overallX = Math.min(
                            100,
                            (opt.overallPct / groupedMaxPct) * 100
                          );
                          return (
                            <div
                              key={g.level}
                              className="grid grid-cols-[130px_1fr_92px] items-center gap-2 text-[11px]"
                            >
                              <span
                                className={[
                                  "truncate",
                                  g.isMissing
                                    ? "italic text-slate-400"
                                    : "text-slate-700",
                                ].join(" ")}
                                title={`${g.level} (n=${g.n})`}
                              >
                                {g.level}
                              </span>
                              <div className="relative h-4 bg-slate-200/60 rounded overflow-hidden">
                                <div
                                  className="h-full rounded-r transition-all"
                                  style={{
                                    width: `${barW}%`,
                                    backgroundColor: groupColor(
                                      gi,
                                      g.isMissing
                                    ),
                                  }}
                                />
                                {/* Overall reference vertical line */}
                                <div
                                  className="absolute top-0 bottom-0 w-px border-l border-dashed border-slate-700/70 pointer-events-none"
                                  style={{ left: `${overallX}%` }}
                                  title={`Overall: ${opt.overallPct.toFixed(1)}%`}
                                />
                              </div>
                              <div className="font-mono text-[10px] text-right whitespace-nowrap">
                                <span className="font-semibold text-slate-800">
                                  {share.toFixed(1)}%
                                </span>{" "}
                                <span
                                  className={
                                    Math.abs(diffPp) >= 5
                                      ? diffPp > 0
                                        ? "text-emerald-700 font-semibold"
                                        : "text-rose-700 font-semibold"
                                      : "text-slate-400"
                                  }
                                >
                                  ({diffPp >= 0 ? "+" : ""}
                                  {diffPp.toFixed(1)})
                                </span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Exact Contingency Table */}
          {showTable && (
            <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
              <div className="px-4 py-2.5 border-b border-slate-200 bg-slate-50/70 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-800">
                  {isMarginal
                    ? "Marginal distribution table (% — aggregate, no microdata N)"
                    : "Marginal distribution table (% and count)"}
                </span>
                <span className="text-[11px] font-mono text-slate-500">
                  {visibleGroups.length} groups × {options.length} options
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-left text-[11px] text-slate-600">
                      <th className="py-2 px-3 font-semibold sticky left-0 bg-slate-50 min-w-[160px]">
                        {data.selectedPersona?.promptLabel}
                      </th>
                      {!isMarginal && (
                        <th className="py-2 px-2.5 font-semibold text-right font-mono">
                          n
                        </th>
                      )}
                      {isScore && (
                        <th className="py-2 px-2.5 font-semibold text-right font-mono">
                          Mean
                        </th>
                      )}
                      {options.map((opt) => (
                        <th
                          key={opt.code}
                          className="py-2 px-2.5 font-semibold text-right min-w-[96px]"
                          title={opt.text}
                        >
                          {truncateLabel(opt.text, 22)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    <tr className="bg-blue-50/40 font-semibold">
                      <td className="py-2 px-3 sticky left-0 bg-blue-50/40 text-slate-900">
                        {isMarginal ? "Group mean" : "Overall"}
                      </td>
                      {!isMarginal && (
                        <td className="py-2 px-2.5 text-right font-mono text-slate-700">
                          {(data.overallN ?? 0).toLocaleString()}
                        </td>
                      )}
                      {isScore && (
                        <td className="py-2 px-2.5 text-right font-mono text-violet-800">
                          {data.overallMean != null
                            ? data.overallMean.toFixed(2)
                            : "—"}
                        </td>
                      )}
                      {options.map((opt) => (
                        <td
                          key={opt.code}
                          className="py-2 px-2.5 text-right font-mono text-slate-800 whitespace-nowrap"
                        >
                          {opt.overallPct.toFixed(1)}%
                          {!isMarginal && (
                            <>
                              {" "}
                              <span className="text-[10px] text-slate-400 font-normal">
                                ({opt.overallCount})
                              </span>
                            </>
                          )}
                        </td>
                      ))}
                    </tr>

                    {visibleGroups.map((g) => (
                      <tr
                        key={g.level}
                        className="hover:bg-slate-50/80 transition-colors"
                      >
                        <td
                          className={[
                            "py-1.5 px-3 sticky left-0 bg-white",
                            g.isMissing
                              ? "italic text-slate-500"
                              : "font-medium text-slate-800",
                          ].join(" ")}
                        >
                          {g.level}
                        </td>
                        {!isMarginal && (
                          <td className="py-1.5 px-2.5 text-right font-mono text-slate-600">
                            {g.n.toLocaleString()}
                          </td>
                        )}
                        {isScore && (
                          <td className="py-1.5 px-2.5 text-right font-mono text-slate-700">
                            {g.meanScore != null ? g.meanScore.toFixed(2) : "—"}
                          </td>
                        )}
                        {options.map((opt, oi) => {
                          const share = g.shares[oi] ?? 0;
                          const diffPp = share - opt.overallPct;
                          return (
                            <td
                              key={opt.code}
                              className={[
                                "py-1.5 px-2.5 text-right font-mono whitespace-nowrap",
                                Math.abs(diffPp) >= 8
                                  ? diffPp > 0
                                    ? "bg-emerald-50/70 text-emerald-900 font-semibold"
                                    : "bg-rose-50/70 text-rose-900 font-semibold"
                                  : "text-slate-700",
                              ].join(" ")}
                              title={`${formatDeltaPp(diffPp)} vs ${isMarginal ? "mean" : "overall"}`}
                            >
                              {share.toFixed(1)}%
                              {!isMarginal && (
                                <>
                                  {" "}
                                  <span className="text-[10px] text-slate-400 font-normal">
                                    ({g.counts[oi]})
                                  </span>
                                </>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
