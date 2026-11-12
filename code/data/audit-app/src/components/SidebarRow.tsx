"use client";

import React, { useEffect, useRef } from "react";
import { ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { getEffectiveType } from "@/lib/triage";
import type { AuditRow, SchemaKind, SidebarSubItem } from "@/lib/types";

interface SidebarRowProps {
  row: AuditRow;
  schemaKind: SchemaKind;
  subItems: SidebarSubItem[];
  overrideStem?: string;
  isSelected: boolean;
  isExpanded: boolean;
  activeSubItemIdx: number;
  onSelect: (rowIdx: number) => void;
  onToggleExpand: (rowKey: string, currentExpanded: boolean) => void;
  onSelectSubItem: (rowIdx: number, subIdx: number) => void;
}

const EMPTY_SUB_ITEMS: SidebarSubItem[] = [];

export const SidebarRow = React.memo(function SidebarRow({
  row,
  schemaKind,
  subItems = EMPTY_SUB_ITEMS,
  overrideStem,
  isSelected,
  isExpanded = false,
  activeSubItemIdx = -1,
  onSelect,
  onToggleExpand,
  onSelectSubItem,
}: SidebarRowProps) {
  const ref = useRef<HTMLDivElement>(null);
  const activeSubRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (isSelected) {
      if (isExpanded && activeSubItemIdx >= 0 && activeSubRef.current) {
        activeSubRef.current.scrollIntoView({ block: "nearest" });
      } else if (ref.current) {
        ref.current.scrollIntoView({ block: "nearest" });
      }
    }
  }, [isSelected, isExpanded, activeSubItemIdx]);

  const d = row.data;
  const title =
    schemaKind === "question_audit"
      ? (d.study ? `${d.study} · ` : "") + d.QuestionID
      : d.v + (d.source_q ? ` · ${d.source_q}` : "");

  const effStem =
    overrideStem !== undefined
      ? overrideStem
      : schemaKind === "question_audit"
        ? d.QuestionText
        : d.question;

  const snippet =
    schemaKind === "question_audit"
      ? effStem || d.Rows_preview || d.Context_preview || "(empty stem)"
      : effStem || d.data_label || "(empty stem)";

  const eff = getEffectiveType(schemaKind, d);

  let badgeClass = "";
  let badgeLabel = "";
  if (row.isFormatSkipped && eff === "exclude") {
    badgeClass = "bg-slate-200/80 text-slate-500 border-slate-300/70";
    badgeLabel = row.formatSkipLabel || "SKIP";
  } else if (row.isFormatSkipped && eff === "noul") {
    badgeClass = "bg-slate-200/80 text-slate-500 border-slate-300/70";
    badgeLabel = "NOUL";
  } else if (eff === "score") {
    badgeClass = "bg-violet-100 text-violet-800 border-violet-200";
    badgeLabel = "score";
  } else if (eff === "choice") {
    badgeClass = "bg-blue-100 text-blue-800 border-blue-200";
    badgeLabel = "choice";
  } else {
    badgeClass = "bg-rose-100 text-rose-800 border-rose-200";
    badgeLabel = "EXCLUDED";
  }

  const isDirty = Boolean(row.isDirty);
  const rowDimmed =
    !isDirty && row.isFormatSkipped && eff !== "score" && eff !== "choice";
  const hasGroup = subItems.length > 1;
  const totalSubItemCount =
    schemaKind === "question_audit" &&
    d.n_subitems &&
    Number(d.n_subitems) > subItems.length
      ? Number(d.n_subitems)
      : subItems.length;

  const handleMainClick = () => {
    if (!hasGroup) {
      onSelect(row.rowIdx);
      return;
    }
    if (!isSelected) {
      onSelect(row.rowIdx);
      if (!isExpanded) {
        onToggleExpand(row.rowKey, false);
      }
    } else {
      onToggleExpand(row.rowKey, isExpanded);
    }
  };

  return (
    <div ref={ref} className="border-b border-slate-100">
      <div
        className={[
          "w-full flex items-stretch transition-colors",
          isSelected
            ? "bg-blue-50/90 border-l-[3px] border-l-blue-600"
            : isDirty
              ? "bg-orange-50/45 border-l-[3px] border-l-orange-400 hover:bg-orange-50/80"
              : rowDimmed
                ? "bg-slate-50/70 opacity-55 hover:opacity-85 hover:bg-slate-100/70"
                : "bg-white hover:bg-slate-50",
        ].join(" ")}
      >
        {hasGroup && (
          <button
            type="button"
            aria-label={isExpanded ? "Collapse sub-items" : "Expand sub-items"}
            title={
              isExpanded
                ? `Collapse ${totalSubItemCount} sub-items`
                : `Expand ${totalSubItemCount} sub-items`
            }
            onClick={() => onToggleExpand(row.rowKey, isExpanded)}
            className={[
              "shrink-0 flex items-start pt-2.5 px-1.5 text-slate-400 hover:text-slate-700 cursor-pointer",
              isSelected || isDirty ? "pl-[6px]" : "pl-[9px]",
            ].join(" ")}
          >
            <ChevronRight
              className={[
                "w-3.5 h-3.5 transition-transform duration-150",
                isExpanded ? "rotate-90 text-slate-700" : "",
              ].join(" ")}
            />
          </button>
        )}

        <button
          type="button"
          onClick={handleMainClick}
          className={[
            "flex-1 min-w-0 text-left py-2.5 pr-3 cursor-pointer",
            hasGroup
              ? "pl-0.5"
              : isSelected || isDirty
                ? "pl-[9px]"
                : "pl-3",
          ].join(" ")}
        >
          <div className="flex items-center gap-1.5 mb-1">
            {isDirty ? (
              <span
                className="w-2 h-2 rounded-full bg-orange-500 shrink-0"
                title={`Unsaved changes: ${(row.dirtyFields ?? []).join(", ")}`}
              />
            ) : rowDimmed ? (
              <span
                className="w-2 h-2 rounded-full bg-slate-300 shrink-0"
                title={`Format-skipped: ${row.formatSkipReason}`}
              />
            ) : (
              <span
                className={[
                  "w-2 h-2 rounded-full shrink-0",
                  row.reviewed
                    ? "bg-emerald-500"
                    : row.isSuspect
                      ? "bg-amber-400"
                      : "bg-slate-300",
                ].join(" ")}
                title={
                  row.reviewed
                    ? "Reviewed"
                    : row.isSuspect
                      ? `Suspect: ${row.suspectReasons.join(", ")}`
                      : "Unreviewed"
                }
              />
            )}
            <span
              className={[
                "text-xs font-mono truncate",
                rowDimmed
                  ? "text-slate-400 font-normal"
                  : "text-slate-800 font-semibold",
              ].join(" ")}
            >
              {title}
            </span>
            <div className="flex-1" />
            {hasGroup && (
              <Badge
                variant="outline"
                className={[
                  "h-4.5 px-1.5 py-0 text-[10px] font-mono rounded-sm shrink-0",
                  isExpanded
                    ? "bg-violet-50 text-violet-800 border-violet-200"
                    : "bg-slate-100 text-slate-600 border-slate-200",
                ].join(" ")}
                title={`${totalSubItemCount} sub-items — click to ${isExpanded ? "collapse" : "expand"}`}
              >
                {totalSubItemCount} items {isExpanded ? "▾" : "▸"}
              </Badge>
            )}
            {row.hasOverride && (
              <Badge
                variant="outline"
                className="h-4.5 px-1.5 py-0 text-[10px] font-semibold rounded-sm bg-cyan-50 text-cyan-800 border-cyan-200 shrink-0 uppercase tracking-tight"
                title="Overridden in question_overrides.json"
              >
                OVERRIDE
              </Badge>
            )}
            {isDirty && (
              <Badge
                variant="outline"
                className="h-4.5 px-1.5 py-0 text-[10px] font-bold rounded-sm bg-orange-100 text-orange-800 border-orange-300 shrink-0 uppercase tracking-tight"
                title={`Unsaved edits (${(row.dirtyFields ?? []).join(", ")}) — click Confirm & Next or Save to write to CSV`}
              >
                UNSAVED
              </Badge>
            )}
            {!rowDimmed && row.isSuspect && !row.reviewed && (
              <Badge
                variant="outline"
                className="h-4.5 px-1.5 py-0 text-[10px] font-semibold rounded-sm bg-amber-100 text-amber-800 border-amber-200 shrink-0"
                title={row.suspectReasons.join(" • ")}
              >
                ?
              </Badge>
            )}
            <Badge
              variant="outline"
              className={`h-4.5 px-1.5 py-0 text-[10px] font-semibold rounded-sm shrink-0 uppercase tracking-tight ${badgeClass}`}
            >
              {badgeLabel}
            </Badge>
          </div>
          <div
            className={[
              "text-xs line-clamp-2 leading-snug",
              rowDimmed ? "text-slate-400 italic" : "text-slate-600",
            ].join(" ")}
          >
            {snippet}
          </div>
          {(row.tags?.length ?? 0) > 0 && (
            <div className="flex items-center gap-1 flex-wrap mt-1">
              {(row.tags ?? []).slice(0, 3).map((tag) => (
                <Badge
                  key={tag}
                  variant="outline"
                  className="h-4 px-1.5 py-0 text-[10px] font-mono rounded-sm bg-indigo-50 text-indigo-700 border-indigo-200/90 max-w-[140px] truncate"
                  title={`#${tag}`}
                >
                  #{tag}
                </Badge>
              ))}
              {(row.tags?.length ?? 0) > 3 && (
                <Badge
                  variant="outline"
                  className="h-4 px-1 py-0 text-[10px] font-mono rounded-sm bg-indigo-50/70 text-indigo-600 border-indigo-200/80"
                  title={(row.tags ?? []).map((t) => `#${t}`).join(", ")}
                >
                  +{(row.tags?.length ?? 0) - 3}
                </Badge>
              )}
            </div>
          )}
        </button>
      </div>

      {/* Collapsible Sub-items Tree */}
      {hasGroup && isExpanded && (
        <div className="bg-slate-50/80 pl-5 pr-2.5 py-1.5 border-t border-slate-100/90">
          <div className="border-l-2 border-slate-200/90 pl-2 space-y-0.5">
            {subItems.map((sub) => {
              const isActiveSub = isSelected && activeSubItemIdx === sub.idx;
              return (
                <button
                  key={`${sub.idx}-${sub.id}`}
                  ref={isActiveSub ? activeSubRef : undefined}
                  type="button"
                  onClick={() => onSelectSubItem(row.rowIdx, sub.idx)}
                  title={`${sub.id}: ${sub.label}`}
                  className={[
                    "w-full text-left px-2 py-1.5 rounded-md text-[11px] leading-snug transition-colors flex items-start gap-1.5 cursor-pointer",
                    isActiveSub
                      ? "bg-violet-100/80 text-violet-950 font-medium ring-1 ring-violet-300/80"
                      : "text-slate-600 hover:bg-white hover:text-slate-900",
                  ].join(" ")}
                >
                  <span
                    className={[
                      "font-mono text-[10px] shrink-0 pt-0.5 min-w-[20px]",
                      isActiveSub ? "text-violet-700 font-semibold" : "text-slate-400",
                    ].join(" ")}
                  >
                    {sub.idx + 1}.
                  </span>
                  <span className="line-clamp-2 flex-1 min-w-0">
                    {sub.label}
                  </span>
                </button>
              );
            })}
            {totalSubItemCount > subItems.length && (
              <div className="px-2 py-1 text-[10px] italic text-slate-400">
                + {totalSubItemCount - subItems.length} more loop items
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
});
