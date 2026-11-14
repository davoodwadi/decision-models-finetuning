"use client";

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Check,
  Database,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  Save,
  Search,
  XIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  classifyAuditRow,
  describeDirtyOverrideFields,
  getSidebarSubItems,
  isRowIncluded,
  normalizeQuestionOverride,
  normalizeTagList,
  overridesEqual,
  resolveContextText,
  tagsEqual,
} from "@/lib/triage";
import type {
  AuditRow,
  DatasetInfo,
  DatasetPayload,
  InitialViewState,
  QuestionOverride,
  SidebarSubItem,
  TriageMode,
} from "@/lib/types";
import { DetailPane } from "./DetailPane";
import { MultiSelectFilter } from "./MultiSelectFilter";
import { SidebarRow } from "./SidebarRow";

const LAST_DS_STORAGE_KEY = "audit-ui-last-ds:v1";

function dsStorageKey(dsId: string): string {
  return `audit-ui-ds:v1:${dsId}`;
}

interface StoredDatasetView {
  rowKey?: string;
  triage?: TriageMode;
  showFormatSkipped?: boolean;
  studyFilter?: string;
  familyFilter?: string[];
  tagFilter?: string[];
  search?: string;
  draftsByRowKey?: Record<string, Record<string, string>>;
  draftOverridesByRowKey?: Record<string, QuestionOverride | null>;
  draftTagsByRowKey?: Record<string, string[]>;
}

function readStoredDatasetView(dsId: string): StoredDatasetView | null {
  try {
    const raw = window.localStorage.getItem(dsStorageKey(dsId));
    if (!raw) return null;
    return JSON.parse(raw) as StoredDatasetView;
  } catch {
    return null;
  }
}

function hydrateDraftsForPayload(
  dsPayload: DatasetPayload,
  draftsByRowKey?: Record<string, Record<string, string>>
): Record<number, Record<string, string>> {
  if (!draftsByRowKey) return {};
  const byKey = new Map(dsPayload.rows.map((r) => [r.rowKey, r]));
  const out: Record<number, Record<string, string>> = {};
  for (const [rowKey, patch] of Object.entries(draftsByRowKey)) {
    const savedRow = byKey.get(rowKey);
    if (!savedRow || !patch) continue;
    const diff: Record<string, string> = {};
    for (const col of dsPayload.columns) {
      if (col === "Context_preview") continue;
      if (col in patch && (patch[col] ?? "") !== (savedRow.data[col] ?? "")) {
        diff[col] = patch[col] ?? "";
      }
    }
    if (Object.keys(diff).length > 0) {
      out[savedRow.rowIdx] = diff;
    }
  }
  return out;
}

function hydrateDraftOverridesForPayload(
  dsPayload: DatasetPayload,
  draftOverridesByRowKey?: Record<string, QuestionOverride | null>
): Record<number, QuestionOverride | null> {
  if (!draftOverridesByRowKey) return {};
  const byKey = new Map(dsPayload.rows.map((r) => [r.rowKey, r]));
  const savedOverrides = dsPayload.overrides ?? {};
  const out: Record<number, QuestionOverride | null> = {};
  for (const [rowKey, rawOv] of Object.entries(draftOverridesByRowKey)) {
    const savedRow = byKey.get(rowKey);
    if (!savedRow) continue;
    const norm = normalizeQuestionOverride(
      rawOv,
      dsPayload.schemaKind,
      savedRow.data,
      dsPayload.items[rowKey],
      dsPayload.dsId
    );
    if (!overridesEqual(norm, savedOverrides[rowKey])) {
      out[savedRow.rowIdx] = norm;
    }
  }
  return out;
}

function hydrateDraftTagsForPayload(
  dsPayload: DatasetPayload,
  draftTagsByRowKey?: Record<string, string[]>
): Record<number, string[]> {
  if (!draftTagsByRowKey) return {};
  const byKey = new Map(dsPayload.rows.map((r) => [r.rowKey, r]));
  const savedTags = dsPayload.tags ?? {};
  const out: Record<number, string[]> = {};
  for (const [rowKey, rawList] of Object.entries(draftTagsByRowKey)) {
    const savedRow = byKey.get(rowKey);
    if (!savedRow) continue;
    const norm = normalizeTagList(rawList);
    if (!tagsEqual(norm, savedTags[rowKey])) {
      out[savedRow.rowIdx] = norm;
    }
  }
  return out;
}

function resolveInitialRowIdx(
  dsPayload: DatasetPayload,
  targetRowKey?: string,
  mode?: TriageMode
): number {
  if (targetRowKey) {
    const found = dsPayload.rows.find((r) => r.rowKey === targetRowKey);
    if (found) return found.rowIdx;
  }
  if (!mode || mode === "suspects") {
    const firstSuspect = dsPayload.rows.find((r) => r.isSuspect);
    if (firstSuspect) return firstSuspect.rowIdx;
  }
  const firstSub = dsPayload.rows.find((r) => !r.isFormatSkipped);
  return firstSub ? firstSub.rowIdx : dsPayload.rows[0]?.rowIdx ?? -1;
}

interface AuditWorkspaceProps {
  initialDatasets: DatasetInfo[];
  initialPayload: DatasetPayload;
  initialView?: InitialViewState;
}

const EMPTY_INITIAL_VIEW: InitialViewState = { hasParams: false };

export function AuditWorkspace({
  initialDatasets,
  initialPayload,
  initialView = EMPTY_INITIAL_VIEW,
}: AuditWorkspaceProps) {
  const [datasets] = useState<DatasetInfo[]>(initialDatasets);
  const [payload, setPayload] = useState<DatasetPayload>(initialPayload);
  const [loadingDs, setLoadingDs] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  const [triage, setTriage] = useState<TriageMode>(() => {
    if (initialView.mode) return initialView.mode;
    return initialPayload.rows.some((r) => r.isSuspect)
      ? "suspects"
      : "substantive";
  });
  const [showFormatSkipped, setShowFormatSkipped] = useState<boolean>(
    initialView.skip ?? false
  );
  const [studyFilter, setStudyFilter] = useState(initialView.study ?? "");
  const [familyFilter, setFamilyFilter] = useState<string[]>(
    initialView.families ?? []
  );
  const [tagFilter, setTagFilter] = useState<string[]>(initialView.tags ?? []);
  const [search, setSearch] = useState(initialView.q ?? "");
  const [bulkSepInput, setBulkSepInput] = useState("\\n\\n");
  const [bulkTagInput, setBulkTagInput] = useState("");

  const [selectedRowIdx, setSelectedRowIdx] = useState<number>(() =>
    resolveInitialRowIdx(initialPayload, initialView.rowKey, initialView.mode)
  );
  const [selectedSubItemIdx, setSelectedSubItemIdx] = useState(0);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>(
    {}
  );

  const [saveState, setSaveState] = useState<
    "idle" | "saving" | "saved" | "error"
  >("saved");
  const [saveMessage, setSaveMessage] = useState("All saved to CSV");
  const [draftPatches, setDraftPatches] = useState<
    Record<number, Record<string, string>>
  >({});
  const [draftOverrides, setDraftOverrides] = useState<
    Record<number, QuestionOverride | null>
  >({});
  const [draftTags, setDraftTags] = useState<Record<number, string[]>>({});
  const [draftResetSeq, setDraftResetSeq] = useState(0);
  const [rebuilding, setRebuilding] = useState(false);
  const [rebuildBanner, setRebuildBanner] = useState<string | null>(null);
  const [showBulk, setShowBulk] = useState(false);

  const searchInputRef = useRef<HTMLInputElement>(null);

  const unsavedRowIndices = useMemo(() => {
    const idxSet = new Set<number>();
    for (const k of Object.keys(draftPatches)) idxSet.add(Number(k));
    for (const k of Object.keys(draftOverrides)) idxSet.add(Number(k));
    for (const k of Object.keys(draftTags)) idxSet.add(Number(k));
    return Array.from(idxSet).sort((a, b) => a - b);
  }, [draftPatches, draftOverrides, draftTags]);
  const unsavedTotal = unsavedRowIndices.length;

  useEffect(() => {
    if (unsavedTotal === 0) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [unsavedTotal]);

  const switchDataset = useCallback(
    async (nextDsId: string, force = false) => {
      const pendingCount = new Set([
        ...Object.keys(draftPatches),
        ...Object.keys(draftOverrides),
        ...Object.keys(draftTags),
      ]).size;
      if (!force && nextDsId !== payload.dsId && pendingCount > 0) {
        const ok = window.confirm(
          `You have ${pendingCount} question(s) with unsaved changes. Discard them and switch dataset?`
        );
        if (!ok) return;
        try {
          const prevStored = readStoredDatasetView(payload.dsId);
          if (prevStored) {
            window.localStorage.setItem(
              dsStorageKey(payload.dsId),
              JSON.stringify({
                ...prevStored,
                draftsByRowKey: {},
                draftOverridesByRowKey: {},
                draftTagsByRowKey: {},
              })
            );
          }
        } catch {
          // Ignore storage write errors
        }
      }
      setLoadingDs(true);
      setRebuildBanner(null);
      try {
        const res = await fetch(
          `/api/dataset?ds=${encodeURIComponent(nextDsId)}`,
          { cache: "no-store" }
        );
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Failed to load dataset");
        const nextPayload = data as DatasetPayload;
        const stored = readStoredDatasetView(nextDsId);
        const restoredDrafts = hydrateDraftsForPayload(
          nextPayload,
          stored?.draftsByRowKey
        );
        const restoredOverrides = hydrateDraftOverridesForPayload(
          nextPayload,
          stored?.draftOverridesByRowKey
        );
        const restoredTags = hydrateDraftTagsForPayload(
          nextPayload,
          stored?.draftTagsByRowKey
        );
        const hasAnyDrafts =
          Object.keys(restoredDrafts).length > 0 ||
          Object.keys(restoredOverrides).length > 0 ||
          Object.keys(restoredTags).length > 0;
        const hasSuspects = nextPayload.rows.some((r) => r.isSuspect);
        const defaultMode: TriageMode = hasSuspects
          ? "suspects"
          : "substantive";
        const nextMode: TriageMode =
          stored?.triage === "unsaved" && !hasAnyDrafts
            ? defaultMode
            : (stored?.triage ?? defaultMode);

        setPayload(nextPayload);
        setDraftPatches(restoredDrafts);
        setDraftOverrides(restoredOverrides);
        setDraftTags(restoredTags);
        setDraftResetSeq((n) => n + 1);
        setShowFormatSkipped(stored?.showFormatSkipped ?? false);
        setStudyFilter(stored?.studyFilter ?? "");
        setFamilyFilter(stored?.familyFilter ?? []);
        setTagFilter(stored?.tagFilter ?? []);
        setSearch(stored?.search ?? "");
        setTriage(nextMode);
        setSelectedRowIdx(
          resolveInitialRowIdx(nextPayload, stored?.rowKey, nextMode)
        );
        setSelectedSubItemIdx(0);
        setExpandedGroups({});
        setSaveState("saved");
        setSaveMessage("All saved");
      } catch (err: unknown) {
        setSaveState("error");
        setSaveMessage(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingDs(false);
      }
    },
    [payload.dsId, draftPatches, draftOverrides, draftTags]
  );

  // Restore saved place & unsaved drafts on client mount
  const didMountRestoreRef = useRef(false);
  useEffect(() => {
    if (didMountRestoreRef.current) return;
    didMountRestoreRef.current = true;

    if (initialView.hasParams) {
      const stored = readStoredDatasetView(initialPayload.dsId);
      const restoredDrafts = hydrateDraftsForPayload(
        initialPayload,
        stored?.draftsByRowKey
      );
      const restoredOverrides = hydrateDraftOverridesForPayload(
        initialPayload,
        stored?.draftOverridesByRowKey
      );
      const restoredTags = hydrateDraftTagsForPayload(
        initialPayload,
        stored?.draftTagsByRowKey
      );
      if (
        Object.keys(restoredDrafts).length > 0 ||
        Object.keys(restoredOverrides).length > 0 ||
        Object.keys(restoredTags).length > 0
      ) {
        setDraftPatches(restoredDrafts);
        setDraftOverrides(restoredOverrides);
        setDraftTags(restoredTags);
        setDraftResetSeq((n) => n + 1);
      }
      if (initialView.rowKey) {
        setSelectedRowIdx(
          resolveInitialRowIdx(
            initialPayload,
            initialView.rowKey,
            initialView.mode
          )
        );
      }
      setHydrated(true);
      return;
    }

    let lastDs: string | null = null;
    try {
      lastDs = window.localStorage.getItem(LAST_DS_STORAGE_KEY);
    } catch {
      lastDs = null;
    }
    if (
      lastDs &&
      lastDs !== initialPayload.dsId &&
      datasets.some((d) => d.id === lastDs)
    ) {
      void switchDataset(lastDs, true).finally(() => setHydrated(true));
      return;
    }

    const stored = readStoredDatasetView(initialPayload.dsId);
    if (stored) {
      const restoredDrafts = hydrateDraftsForPayload(
        initialPayload,
        stored.draftsByRowKey
      );
      const restoredOverrides = hydrateDraftOverridesForPayload(
        initialPayload,
        stored.draftOverridesByRowKey
      );
      const restoredTags = hydrateDraftTagsForPayload(
        initialPayload,
        stored.draftTagsByRowKey
      );
      const hasAnyDrafts =
        Object.keys(restoredDrafts).length > 0 ||
        Object.keys(restoredOverrides).length > 0 ||
        Object.keys(restoredTags).length > 0;
      const hasSuspects = initialPayload.rows.some((r) => r.isSuspect);
      const defaultMode: TriageMode = hasSuspects ? "suspects" : "substantive";
      const nextMode: TriageMode =
        stored.triage === "unsaved" && !hasAnyDrafts
          ? defaultMode
          : (stored.triage ?? defaultMode);
      if (hasAnyDrafts) {
        setDraftPatches(restoredDrafts);
        setDraftOverrides(restoredOverrides);
        setDraftTags(restoredTags);
        setDraftResetSeq((n) => n + 1);
      }
      setShowFormatSkipped(stored.showFormatSkipped ?? false);
      setStudyFilter(stored.studyFilter ?? "");
      setFamilyFilter(stored.familyFilter ?? []);
      setTagFilter(stored.tagFilter ?? []);
      setSearch(stored.search ?? "");
      setTriage(nextMode);
      setSelectedRowIdx(
        resolveInitialRowIdx(initialPayload, stored.rowKey, nextMode)
      );
    }
    setHydrated(true);
  }, [datasets, initialPayload, initialView, switchDataset]);

  // Live disk sync: pick up AI/external edits to question_tags.json, overrides, or CSV on focus
  useEffect(() => {
    const syncFromDisk = async () => {
      if (saveState === "saving" || loadingDs) return;
      try {
        const res = await fetch(
          `/api/dataset?ds=${encodeURIComponent(payload.dsId)}`
        );
        if (!res.ok) return;
        const fresh = (await res.json()) as DatasetPayload;
        if (fresh.dsId !== payload.dsId) return;
        setPayload((prev) => {
          if (prev.dsId !== fresh.dsId) return prev;
          if (
            JSON.stringify(prev.tags ?? {}) ===
              JSON.stringify(fresh.tags ?? {}) &&
            JSON.stringify(prev.overrides ?? {}) ===
              JSON.stringify(fresh.overrides ?? {}) &&
            prev.rows.length === fresh.rows.length &&
            prev.rows.every(
              (r, i) =>
                r.reviewed === fresh.rows[i]?.reviewed &&
                JSON.stringify(r.data) === JSON.stringify(fresh.rows[i]?.data)
            )
          ) {
            return prev;
          }
          return fresh;
        });
      } catch {
        // Ignore transient network errors during background sync
      }
    };
    const onFocus = () => {
      void syncFromDisk();
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        void syncFromDisk();
      }
    };
    window.addEventListener("focus", onFocus);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("focus", onFocus);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [payload.dsId, saveState, loadingDs]);

  const effectiveOverridesByRowKey = useMemo(() => {
    const out: Record<string, QuestionOverride | undefined> = {
      ...(payload.overrides ?? {}),
    };
    for (const [idxStr, draftOv] of Object.entries(draftOverrides)) {
      const rk = payload.rows[Number(idxStr)]?.rowKey;
      if (!rk) continue;
      if (draftOv) {
        out[rk] = draftOv;
      } else {
        delete out[rk];
      }
    }
    return out;
  }, [payload.overrides, payload.rows, draftOverrides]);

  const effectiveTagsByRowKey = useMemo(() => {
    const out: Record<string, string[] | undefined> = {
      ...(payload.tags ?? {}),
    };
    for (const [idxStr, nextTags] of Object.entries(draftTags)) {
      const rk = payload.rows[Number(idxStr)]?.rowKey;
      if (!rk) continue;
      if (nextTags.length > 0) {
        out[rk] = nextTags;
      } else {
        delete out[rk];
      }
    }
    return out;
  }, [payload.tags, payload.rows, draftTags]);

  // Merge committed CSV rows + overrides + tags with staged in-memory drafts
  const displayRows = useMemo(() => {
    const savedOverrides = payload.overrides ?? {};
    const savedTags = payload.tags ?? {};
    return payload.rows.map((savedRow) => {
      const draft = draftPatches[savedRow.rowIdx];
      const hasDraftPatch = Boolean(draft && Object.keys(draft).length > 0);
      const hasDraftOv = savedRow.rowIdx in draftOverrides;
      const hasDraftTags = savedRow.rowIdx in draftTags;
      if (!hasDraftPatch && !hasDraftOv && !hasDraftTags) {
        return savedRow.tags !== undefined
          ? savedRow
          : { ...savedRow, tags: savedTags[savedRow.rowKey] ?? [] };
      }
      const nextData = hasDraftPatch
        ? { ...savedRow.data, ...draft }
        : savedRow.data;
      if (
        hasDraftPatch &&
        draft &&
        payload.schemaKind === "question_audit" &&
        "context_source" in draft &&
        payload.columns.includes("Context_preview")
      ) {
        const ctxKey = nextData.study || "__all__";
        nextData.Context_preview = resolveContextText(
          nextData.context_source,
          payload.items[savedRow.rowKey],
          payload.studyContexts[ctxKey]
        );
      }
      const effOv = effectiveOverridesByRowKey[savedRow.rowKey];
      const effTags = effectiveTagsByRowKey[savedRow.rowKey];
      const classified = classifyAuditRow(
        savedRow.rowIdx,
        payload.schemaKind,
        payload.columns,
        nextData,
        savedRow.reviewed,
        payload.items[savedRow.rowKey],
        effOv,
        effTags
      );
      const dirtyFields = [
        ...(draft ? Object.keys(draft) : []),
        ...(hasDraftOv
          ? describeDirtyOverrideFields(
              savedOverrides[savedRow.rowKey],
              draftOverrides[savedRow.rowIdx]
            )
          : []),
        ...(hasDraftTags &&
        !tagsEqual(savedTags[savedRow.rowKey], draftTags[savedRow.rowIdx])
          ? ["tags"]
          : []),
      ];
      return {
        ...classified,
        isDirty: dirtyFields.length > 0,
        dirtyFields,
      };
    });
  }, [
    payload.rows,
    payload.schemaKind,
    payload.columns,
    payload.items,
    payload.studyContexts,
    payload.overrides,
    payload.tags,
    draftPatches,
    draftOverrides,
    draftTags,
    effectiveOverridesByRowKey,
    effectiveTagsByRowKey,
  ]);

  const subItemsByRowKey = useMemo(() => {
    const out: Record<string, SidebarSubItem[]> = {};
    for (const r of payload.rows) {
      const subs = getSidebarSubItems(
        payload.schemaKind,
        r.data,
        payload.items[r.rowKey],
        effectiveOverridesByRowKey[r.rowKey]
      );
      if (subs.length > 1) {
        out[r.rowKey] = subs;
      }
    }
    return out;
  }, [
    payload.rows,
    payload.schemaKind,
    payload.items,
    effectiveOverridesByRowKey,
  ]);

  // Rows matching search + dropdowns (before triage & format-skip toggle)
  const preTriageRows = useMemo(() => {
    const q = search.trim().toLowerCase();
    const wantedTagSet =
      tagFilter.length > 0
        ? new Set(tagFilter.map((t) => t.toLowerCase()))
        : null;
    return displayRows.filter((r) => {
      const d = r.data;
      const rowSection = payload.columns.includes("study")
        ? d.study || ""
        : payload.columns.includes("BlockName")
          ? (d.BlockName || "").trim()
          : "";
      if (studyFilter && rowSection !== studyFilter) return false;
      if (familyFilter.length > 0) {
        const fv =
          payload.schemaKind === "question_audit" ? d.family : d.type;
        if (!familyFilter.includes(fv)) return false;
      }
      if (wantedTagSet) {
        const rowTagLower = (r.tags ?? []).map((t) => t.toLowerCase());
        if (!rowTagLower.some((t) => wantedTagSet.has(t))) return false;
      }
      if (q) {
        const subText = (subItemsByRowKey[r.rowKey] ?? [])
          .map((s) => `${s.id} ${s.label}`)
          .join(" ");
        const ov = effectiveOverridesByRowKey[r.rowKey];
        const ovText = ov
          ? `${ov.stem ?? ""} ${Object.values(ov.options ?? {}).join(" ")}`
          : "";
        const tagText = (r.tags ?? []).map((t) => `#${t} ${t}`).join(" ");
        const hay = `${Object.values(d).join(" ")} ${subText} ${ovText} ${tagText}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [
    displayRows,
    payload.columns,
    payload.schemaKind,
    subItemsByRowKey,
    effectiveOverridesByRowKey,
    studyFilter,
    familyFilter,
    tagFilter,
    search,
  ]);

  const matchesMode = useCallback(
    (r: AuditRow, mode: TriageMode, includeSkipped: boolean): boolean => {
      if (mode === "all") return true;
      if (mode === "unsaved") return Boolean(r.isDirty);
      if (mode === "overridden") return Boolean(r.hasOverride);
      if (mode === "tagged") return (r.tags?.length ?? 0) > 0;
      if (!includeSkipped && r.isFormatSkipped) {
        const effIncluded = isRowIncluded(payload.schemaKind, r.data);
        if (!effIncluded) return false;
      }
      if (mode === "suspects") return r.isSuspect;
      if (mode === "unreviewed") return !r.isFormatSkipped && !r.reviewed;
      if (mode === "substantive") return !r.isFormatSkipped || includeSkipped;
      if (mode === "included") return isRowIncluded(payload.schemaKind, r.data);
      if (mode === "excluded") return !isRowIncluded(payload.schemaKind, r.data);
      return true;
    },
    [payload.schemaKind]
  );

  const tabCounts = useMemo(() => {
    const modes: TriageMode[] = [
      "suspects",
      "unsaved",
      "overridden",
      "tagged",
      "unreviewed",
      "substantive",
      "included",
      "excluded",
      "all",
    ];
    const out = {} as Record<TriageMode, number>;
    for (const m of modes) {
      out[m] = preTriageRows.filter((r) =>
        matchesMode(r, m, showFormatSkipped)
      ).length;
    }
    return out;
  }, [preTriageRows, matchesMode, showFormatSkipped]);

  const formatSkippedCount = useMemo(
    () => preTriageRows.filter((r) => r.isFormatSkipped).length,
    [preTriageRows]
  );

  const filteredRows = useMemo(
    () =>
      preTriageRows.filter((r) => matchesMode(r, triage, showFormatSkipped)),
    [preTriageRows, matchesMode, triage, showFormatSkipped]
  );

  const filteredIndices = useMemo(
    () => filteredRows.map((r) => r.rowIdx),
    [filteredRows]
  );

  // Keep selection valid when filters change (only after initial restoration)
  useEffect(() => {
    if (!hydrated) return;
    if (filteredIndices.length === 0) {
      setSelectedRowIdx(-1);
      return;
    }
    if (!filteredIndices.includes(selectedRowIdx)) {
      setSelectedRowIdx(filteredIndices[0]);
    }
  }, [hydrated, filteredIndices, selectedRowIdx]);

  // Persist current place (URL query + localStorage including unsaved drafts)
  useEffect(() => {
    if (!hydrated) return;
    const activeRowKey =
      selectedRowIdx >= 0 ? payload.rows[selectedRowIdx]?.rowKey : undefined;

    const sp = new URLSearchParams();
    sp.set("ds", payload.dsId);
    if (activeRowKey) sp.set("row", activeRowKey);
    sp.set("mode", triage);
    if (showFormatSkipped) sp.set("skip", "1");
    if (studyFilter) sp.set("study", studyFilter);
    for (const fam of familyFilter) {
      sp.append("family", fam);
    }
    for (const tg of tagFilter) {
      sp.append("tag", tg);
    }
    if (search) sp.set("q", search);

    const nextUrl = `${window.location.pathname}?${sp.toString()}`;
    if (`${window.location.pathname}${window.location.search}` !== nextUrl) {
      window.history.replaceState(null, "", nextUrl);
    }

    try {
      const draftsByRowKey: Record<string, Record<string, string>> = {};
      for (const [idxStr, patch] of Object.entries(draftPatches)) {
        const rk = payload.rows[Number(idxStr)]?.rowKey;
        if (rk && Object.keys(patch).length > 0) {
          draftsByRowKey[rk] = patch;
        }
      }
      const draftOverridesByRowKey: Record<string, QuestionOverride | null> = {};
      for (const [idxStr, ov] of Object.entries(draftOverrides)) {
        const rk = payload.rows[Number(idxStr)]?.rowKey;
        if (rk) {
          draftOverridesByRowKey[rk] = ov;
        }
      }
      const draftTagsByRowKey: Record<string, string[]> = {};
      for (const [idxStr, tList] of Object.entries(draftTags)) {
        const rk = payload.rows[Number(idxStr)]?.rowKey;
        if (rk) {
          draftTagsByRowKey[rk] = tList;
        }
      }
      const snapshot: StoredDatasetView = {
        rowKey: activeRowKey,
        triage,
        showFormatSkipped,
        studyFilter,
        familyFilter,
        tagFilter,
        search,
        draftsByRowKey,
        draftOverridesByRowKey,
        draftTagsByRowKey,
      };
      window.localStorage.setItem(LAST_DS_STORAGE_KEY, payload.dsId);
      window.localStorage.setItem(
        dsStorageKey(payload.dsId),
        JSON.stringify(snapshot)
      );
    } catch {
      // Ignore localStorage quota/privacy errors
    }
  }, [
    hydrated,
    payload.dsId,
    payload.rows,
    selectedRowIdx,
    triage,
    showFormatSkipped,
    studyFilter,
    familyFilter,
    tagFilter,
    search,
    draftPatches,
    draftOverrides,
    draftTags,
  ]);

  const hasStudyCol = payload.columns.includes("study");
  const hasBlockCol = !hasStudyCol && payload.columns.includes("BlockName");

  // Dropdown option counts
  const studyOptions = useMemo(() => {
    if (!hasStudyCol && !hasBlockCol) return [];
    const counts = new Map<string, number>();
    for (const r of payload.rows) {
      const s = hasStudyCol
        ? r.data.study || ""
        : (r.data.BlockName || "").trim();
      if (!s) continue;
      counts.set(s, (counts.get(s) ?? 0) + 1);
    }
    return Array.from(counts.entries());
  }, [hasStudyCol, hasBlockCol, payload.rows]);

  const familyOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of payload.rows) {
      const k =
        payload.schemaKind === "question_audit" ? r.data.family : r.data.type;
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
  }, [payload.rows, payload.schemaKind]);

  const tagOptions = useMemo(() => {
    const counts = new Map<string, number>();
    const canonicalCase = new Map<string, string>();
    for (const r of displayRows) {
      for (const tg of r.tags ?? []) {
        const lower = tg.toLowerCase();
        const canon = canonicalCase.get(lower) ?? tg;
        canonicalCase.set(lower, canon);
        counts.set(canon, (counts.get(canon) ?? 0) + 1);
      }
    }
    for (const sel of tagFilter) {
      const lower = sel.toLowerCase();
      if (!canonicalCase.has(lower)) {
        canonicalCase.set(lower, sel);
        counts.set(sel, 0);
      }
    }
    return Array.from(counts.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
  }, [displayRows, tagFilter]);

  // Stage edits in memory without writing to CSV until confirmed/saved
  const stageRowsPatch = useCallback(
    (rowIndices: number[], patch: Record<string, string>) => {
      if (rowIndices.length === 0 || Object.keys(patch).length === 0) return;
      setDraftPatches((prev) => {
        const next = { ...prev };
        for (const rowIdx of rowIndices) {
          const savedRow = payload.rows[rowIdx];
          if (!savedRow) continue;
          const merged = {
            ...savedRow.data,
            ...(prev[rowIdx] ?? {}),
            ...patch,
          };
          if (payload.schemaKind === "question_audit") {
            if (merged.include === "False") {
              merged.kev_type = "";
            }
            if ("context_source" in patch && merged.context_source !== undefined) {
              merged.context_source = merged.context_source.trim();
            }
          }
          const diff: Record<string, string> = {};
          for (const col of payload.columns) {
            if (col === "Context_preview") continue;
            const candidateVal = merged[col] ?? "";
            const savedVal = savedRow.data[col] ?? "";
            if (candidateVal !== savedVal) {
              diff[col] = candidateVal;
            }
          }
          if (Object.keys(diff).length > 0) {
            next[rowIdx] = diff;
          } else {
            delete next[rowIdx];
          }
        }
        return next;
      });
    },
    [payload.rows, payload.schemaKind, payload.columns]
  );

  const handleStageSingleRow = useCallback(
    (rowIdx: number, patch: Record<string, string>) => {
      stageRowsPatch([rowIdx], patch);
    },
    [stageRowsPatch]
  );

  const handleStageOverride = useCallback(
    (rowIdx: number, nextOverride: QuestionOverride | null) => {
      const savedRow = payload.rows[rowIdx];
      if (!savedRow) return;
      const mergedData = {
        ...savedRow.data,
        ...(draftPatches[rowIdx] ?? {}),
      };
      const norm = normalizeQuestionOverride(
        nextOverride,
        payload.schemaKind,
        mergedData,
        payload.items[savedRow.rowKey],
        payload.dsId
      );
      const savedOv = (payload.overrides ?? {})[savedRow.rowKey];
      setDraftOverrides((prev) => {
        if (overridesEqual(norm, savedOv)) {
          if (!(rowIdx in prev)) return prev;
          const next = { ...prev };
          delete next[rowIdx];
          return next;
        }
        return { ...prev, [rowIdx]: norm };
      });
    },
    [
      payload.dsId,
      payload.rows,
      payload.schemaKind,
      payload.items,
      payload.overrides,
      draftPatches,
    ]
  );

  const handleStageTags = useCallback(
    (rowIdx: number, nextTags: string[]) => {
      const savedRow = payload.rows[rowIdx];
      if (!savedRow) return;
      const norm = normalizeTagList(nextTags);
      const savedList = (payload.tags ?? {})[savedRow.rowKey] ?? [];
      setDraftTags((prev) => {
        if (tagsEqual(norm, savedList)) {
          if (!(rowIdx in prev)) return prev;
          const next = { ...prev };
          delete next[rowIdx];
          return next;
        }
        return { ...prev, [rowIdx]: norm };
      });
    },
    [payload.rows, payload.tags]
  );

  const stageBulkAddTags = useCallback(
    (rowIndices: number[], incomingRaw: string) => {
      const toAdd = normalizeTagList(incomingRaw.split(","));
      if (rowIndices.length === 0 || toAdd.length === 0) return;
      setDraftTags((prev) => {
        const next = { ...prev };
        const savedMap = payload.tags ?? {};
        for (const rowIdx of rowIndices) {
          const savedRow = payload.rows[rowIdx];
          if (!savedRow) continue;
          const savedList = savedMap[savedRow.rowKey] ?? [];
          const base = rowIdx in next ? next[rowIdx] : savedList;
          const merged = normalizeTagList([...base, ...toAdd]);
          if (tagsEqual(merged, savedList)) {
            delete next[rowIdx];
          } else {
            next[rowIdx] = merged;
          }
        }
        return next;
      });
      setBulkTagInput("");
    },
    [payload.rows, payload.tags]
  );

  const handleDiscardRow = useCallback((rowIdx: number) => {
    setDraftPatches((prev) => {
      if (!(rowIdx in prev)) return prev;
      const next = { ...prev };
      delete next[rowIdx];
      return next;
    });
    setDraftOverrides((prev) => {
      if (!(rowIdx in prev)) return prev;
      const next = { ...prev };
      delete next[rowIdx];
      return next;
    });
    setDraftTags((prev) => {
      if (!(rowIdx in prev)) return prev;
      const next = { ...prev };
      delete next[rowIdx];
      return next;
    });
    setDraftResetSeq((n) => n + 1);
  }, []);

  const handleDiscardAll = useCallback(() => {
    setDraftPatches({});
    setDraftOverrides({});
    setDraftTags({});
    setDraftResetSeq((n) => n + 1);
  }, []);

  // Commit explicit entries to CSV, question_overrides.json, question_tags.json, and/or .review_progress.json
  const commitRowEntries = useCallback(
    async (
      entries: Array<{
        rowIdx: number;
        patch: Record<string, string>;
        override?: QuestionOverride | null;
        tags?: string[] | null;
        reviewed?: boolean;
      }>
    ) => {
      if (entries.length === 0) return;
      const wroteCsv = entries.some((e) => Object.keys(e.patch).length > 0);
      const wroteOverrides = entries.some((e) => e.override !== undefined);
      const wroteTags = entries.some((e) => e.tags !== undefined);
      setSaveState("saving");
      setSaveMessage(
        wroteCsv && (wroteOverrides || wroteTags)
          ? "Saving changes..."
          : wroteOverrides
            ? "Saving overrides..."
            : wroteTags
              ? "Saving tags..."
              : wroteCsv
                ? "Saving to CSV..."
                : "Saving review..."
      );

      const entryByIdx = new Map(entries.map((e) => [e.rowIdx, e]));

      // Optimistic commit on client
      setPayload((prev) => {
        const nextOverrides = { ...(prev.overrides ?? {}) };
        const nextTagsMap = { ...(prev.tags ?? {}) };
        for (const e of entries) {
          const rk = prev.rows[e.rowIdx]?.rowKey;
          if (!rk) continue;
          if (e.override !== undefined) {
            if (e.override) {
              nextOverrides[rk] = e.override;
            } else {
              delete nextOverrides[rk];
            }
          }
          if (e.tags !== undefined) {
            const cleanTags = normalizeTagList(e.tags);
            if (cleanTags.length > 0) {
              nextTagsMap[rk] = cleanTags;
            } else {
              delete nextTagsMap[rk];
            }
          }
        }
        const nextRows = prev.rows.map((r) => {
          const entry = entryByIdx.get(r.rowIdx);
          if (!entry) return r;
          const nextData = { ...r.data, ...entry.patch };
          if (prev.schemaKind === "question_audit") {
            if (nextData.include === "False") {
              nextData.kev_type = "";
            }
            if (
              "context_source" in entry.patch &&
              prev.columns.includes("Context_preview")
            ) {
              const ctxKey = nextData.study || "__all__";
              nextData.Context_preview = resolveContextText(
                nextData.context_source,
                prev.items[r.rowKey],
                prev.studyContexts[ctxKey]
              );
            }
          }
          const nextRev =
            entry.reviewed !== undefined ? entry.reviewed : r.reviewed;
          return classifyAuditRow(
            r.rowIdx,
            prev.schemaKind,
            prev.columns,
            nextData,
            nextRev,
            prev.items[r.rowKey],
            nextOverrides[r.rowKey],
            nextTagsMap[r.rowKey]
          );
        });
        return {
          ...prev,
          rows: nextRows,
          overrides: nextOverrides,
          tags: nextTagsMap,
        };
      });

      if (wroteCsv) {
        setDraftPatches((prev) => {
          const next = { ...prev };
          for (const e of entries) {
            if (Object.keys(e.patch).length > 0) {
              delete next[e.rowIdx];
            }
          }
          return next;
        });
      }
      if (wroteOverrides) {
        setDraftOverrides((prev) => {
          const next = { ...prev };
          for (const e of entries) {
            if (e.override !== undefined) {
              delete next[e.rowIdx];
            }
          }
          return next;
        });
      }
      if (wroteTags) {
        setDraftTags((prev) => {
          const next = { ...prev };
          for (const e of entries) {
            if (e.tags !== undefined) {
              delete next[e.rowIdx];
            }
          }
          return next;
        });
      }

      try {
        const res = await fetch("/api/update", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ds: payload.dsId,
            entries,
          }),
        });
        const json = await res.json();
        if (!res.ok) throw new Error(json.error || "Failed to save");
        const updatedList = json.updated as AuditRow[];
        const serverOverrides = (json.overrides ?? {}) as Record<
          string,
          QuestionOverride
        >;
        const serverTags = (json.tags ?? {}) as Record<string, string[]>;
        const byIdx = new Map(updatedList.map((u) => [u.rowIdx, u]));
        setPayload((prev) => ({
          ...prev,
          rows: prev.rows.map((r) => byIdx.get(r.rowIdx) ?? r),
          overrides: serverOverrides,
          tags: serverTags,
        }));
        setSaveState("saved");
        setSaveMessage(
          wroteCsv && (wroteOverrides || wroteTags)
            ? "Saved CSV, overrides & tags"
            : wroteOverrides && wroteTags
              ? "Saved overrides & tags"
              : wroteOverrides
                ? "Saved to question_overrides.json"
                : wroteTags
                  ? "Saved to question_tags.json"
                  : wroteCsv
                    ? entries.length > 1
                      ? `Saved ${entries.length} rows to CSV`
                      : "Saved to CSV"
                    : "Review progress saved"
        );
      } catch (err: unknown) {
        // Restore patches, overrides & tags back into drafts so edits are never lost on failure
        if (wroteCsv) {
          setDraftPatches((prev) => {
            const restored = { ...prev };
            for (const e of entries) {
              if (Object.keys(e.patch).length > 0) {
                restored[e.rowIdx] = {
                  ...e.patch,
                  ...(restored[e.rowIdx] ?? {}),
                };
              }
            }
            return restored;
          });
        }
        if (wroteOverrides) {
          setDraftOverrides((prev) => {
            const restored = { ...prev };
            for (const e of entries) {
              if (e.override !== undefined && !(e.rowIdx in restored)) {
                restored[e.rowIdx] = e.override;
              }
            }
            return restored;
          });
        }
        if (wroteTags) {
          setDraftTags((prev) => {
            const restored = { ...prev };
            for (const e of entries) {
              if (e.tags !== undefined && !(e.rowIdx in restored)) {
                restored[e.rowIdx] = e.tags ?? [];
              }
            }
            return restored;
          });
        }
        setSaveState("error");
        setSaveMessage(err instanceof Error ? err.message : String(err));
      }
    },
    [payload.dsId]
  );

  const handleSaveSingleRow = useCallback(
    (rowIdx: number) => {
      const pendingPatch = draftPatches[rowIdx] ?? {};
      const hasOv = rowIdx in draftOverrides;
      const hasTg = rowIdx in draftTags;
      if (Object.keys(pendingPatch).length === 0 && !hasOv && !hasTg) return;
      void commitRowEntries([
        {
          rowIdx,
          patch: pendingPatch,
          ...(hasOv ? { override: draftOverrides[rowIdx] } : {}),
          ...(hasTg ? { tags: draftTags[rowIdx] } : {}),
        },
      ]);
    },
    [draftPatches, draftOverrides, draftTags, commitRowEntries]
  );

  const handleSaveAllUnsaved = useCallback(() => {
    const entries = unsavedRowIndices.map((rowIdx) => ({
      rowIdx,
      patch: draftPatches[rowIdx] ?? {},
      ...(rowIdx in draftOverrides ? { override: draftOverrides[rowIdx] } : {}),
      ...(rowIdx in draftTags ? { tags: draftTags[rowIdx] } : {}),
      reviewed: true,
    }));
    if (entries.length === 0) return;
    void commitRowEntries(entries);
  }, [unsavedRowIndices, draftPatches, draftOverrides, draftTags, commitRowEntries]);

  const handleToggleReviewed = useCallback(
    (rowIdx: number, nextReviewed: boolean) => {
      void commitRowEntries([{ rowIdx, patch: {}, reviewed: nextReviewed }]);
    },
    [commitRowEntries]
  );

  const posInFiltered = filteredIndices.indexOf(selectedRowIdx);

  const handleSelectRow = useCallback((rowIdx: number) => {
    setSelectedRowIdx(rowIdx);
    setSelectedSubItemIdx(0);
  }, []);

  const handleToggleGroupExpand = useCallback(
    (rowKey: string, currentExpanded: boolean) => {
      setExpandedGroups((prev) => ({
        ...prev,
        [rowKey]: !currentExpanded,
      }));
    },
    []
  );

  const handleSelectSubItem = useCallback((rowIdx: number, subIdx: number) => {
    setSelectedRowIdx(rowIdx);
    setSelectedSubItemIdx(subIdx);
  }, []);

  const goPrev = useCallback(() => {
    if (posInFiltered > 0) {
      setSelectedRowIdx(filteredIndices[posInFiltered - 1]);
      setSelectedSubItemIdx(0);
    }
  }, [posInFiltered, filteredIndices]);

  const goNext = useCallback(() => {
    if (posInFiltered >= 0 && posInFiltered < filteredIndices.length - 1) {
      setSelectedRowIdx(filteredIndices[posInFiltered + 1]);
      setSelectedSubItemIdx(0);
    }
  }, [posInFiltered, filteredIndices]);

  const handleConfirmAndNext = useCallback(() => {
    if (selectedRowIdx < 0) return;
    const nextIdx =
      posInFiltered >= 0 && posInFiltered < filteredIndices.length - 1
        ? filteredIndices[posInFiltered + 1]
        : selectedRowIdx;
    const pendingPatch = draftPatches[selectedRowIdx] ?? {};
    const hasOv = selectedRowIdx in draftOverrides;
    const hasTg = selectedRowIdx in draftTags;
    void commitRowEntries([
      {
        rowIdx: selectedRowIdx,
        patch: pendingPatch,
        ...(hasOv ? { override: draftOverrides[selectedRowIdx] } : {}),
        ...(hasTg ? { tags: draftTags[selectedRowIdx] } : {}),
        reviewed: true,
      },
    ]);
    if (nextIdx !== selectedRowIdx) {
      setSelectedRowIdx(nextIdx);
      setSelectedSubItemIdx(0);
    }
  }, [
    selectedRowIdx,
    posInFiltered,
    filteredIndices,
    draftPatches,
    draftOverrides,
    draftTags,
    commitRowEntries,
  ]);

  // Global keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        if (
          selectedRowIdx >= 0 &&
          (draftPatches[selectedRowIdx] ||
            selectedRowIdx in draftOverrides ||
            selectedRowIdx in draftTags)
        ) {
          handleSaveSingleRow(selectedRowIdx);
        }
        return;
      }

      if (e.defaultPrevented) return;
      const targetEl = e.target as HTMLElement | null;
      const tag = targetEl?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
        if (e.key === "Escape") {
          targetEl?.blur();
        }
        return;
      }
      if (
        targetEl?.closest(
          '[data-slot="popover-content"], [data-slot="select-content"], [data-slot="select-trigger"], [data-slot="popover-trigger"], [data-slot="checkbox"], [role="listbox"], [role="dialog"]'
        )
      ) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === "/") {
        e.preventDefault();
        searchInputRef.current?.focus();
        return;
      }
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        goNext();
        return;
      }
      if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        goPrev();
        return;
      }
      if (selectedRowIdx < 0) return;

      if (e.key === "s") {
        e.preventDefault();
        if (payload.schemaKind === "question_audit") {
          handleStageSingleRow(selectedRowIdx, {
            include: "True",
            kev_type: "score",
          });
        } else {
          handleStageSingleRow(selectedRowIdx, { type: "score" });
        }
      } else if (e.key === "c") {
        e.preventDefault();
        if (payload.schemaKind === "question_audit") {
          handleStageSingleRow(selectedRowIdx, {
            include: "True",
            kev_type: "choice",
          });
        } else {
          handleStageSingleRow(selectedRowIdx, { type: "choice" });
        }
      } else if (e.key === "x") {
        e.preventDefault();
        if (payload.schemaKind === "question_audit") {
          handleStageSingleRow(selectedRowIdx, {
            include: "False",
            kev_type: "",
          });
        } else {
          handleStageSingleRow(selectedRowIdx, { type: "exclude" });
        }
      } else if (e.key === "Escape") {
        if (
          draftPatches[selectedRowIdx] ||
          selectedRowIdx in draftOverrides ||
          selectedRowIdx in draftTags
        ) {
          e.preventDefault();
          handleDiscardRow(selectedRowIdx);
        }
      } else if (e.key === " ") {
        e.preventDefault();
        const cur = payload.rows[selectedRowIdx];
        if (cur) {
          handleToggleReviewed(selectedRowIdx, !cur.reviewed);
        }
      } else if (e.key === "Enter") {
        e.preventDefault();
        handleConfirmAndNext();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    goNext,
    goPrev,
    selectedRowIdx,
    draftPatches,
    draftOverrides,
    draftTags,
    payload.schemaKind,
    payload.rows,
    handleStageSingleRow,
    handleSaveSingleRow,
    handleDiscardRow,
    handleToggleReviewed,
    handleConfirmAndNext,
  ]);

  const handleRebuild = async () => {
    if (unsavedTotal > 0) {
      setSaveState("error");
      setSaveMessage(
        `Save or discard ${unsavedTotal} unsaved question(s) before rebuilding Parquet`
      );
      setTriage("unsaved");
      return;
    }
    setRebuilding(true);
    setRebuildBanner(null);
    try {
      const res = await fetch("/api/rebuild", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ds: payload.dsId }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        throw new Error(json.stderr || json.error || "Notebook execution failed");
      }
      setRebuildBanner(`Parquet rebuilt: ${json.summary}`);
      await switchDataset(payload.dsId, true);
    } catch (err: unknown) {
      setSaveState("error");
      setSaveMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setRebuilding(false);
    }
  };

  // Header stats
  const substantiveTotal = displayRows.filter((r) => !r.isFormatSkipped).length;
  const substantiveReviewed = displayRows.filter(
    (r) => !r.isFormatSkipped && r.reviewed
  ).length;
  const suspectsRemaining = displayRows.filter(
    (r) => r.isSuspect && !r.reviewed
  ).length;
  const includedTotal = displayRows.filter((r) =>
    isRowIncluded(payload.schemaKind, r.data)
  ).length;

  const currentRow =
    selectedRowIdx >= 0 ? displayRows[selectedRowIdx] : undefined;
  const currentSavedRow =
    selectedRowIdx >= 0 ? payload.rows[selectedRowIdx] : undefined;
  const currentDetail = currentRow
    ? payload.items[currentRow.rowKey]
    : undefined;
  const currentStudyCandidates = currentRow
    ? payload.studyContexts[currentRow.data.study || "__all__"]
    : undefined;
  const currentSavedOverride = currentSavedRow
    ? (payload.overrides ?? {})[currentSavedRow.rowKey]
    : undefined;
  const currentEffectiveOverride = currentRow
    ? effectiveOverridesByRowKey[currentRow.rowKey]
    : undefined;
  const currentSavedTags = currentSavedRow
    ? (payload.tags ?? {})[currentSavedRow.rowKey] ?? []
    : [];

  const handleFilterByTag = useCallback((tag: string) => {
    setTagFilter((prev) => {
      if (prev.some((t) => t.toLowerCase() === tag.toLowerCase())) {
        return prev;
      }
      return [...prev, tag];
    });
  }, []);

  return (
    <div className="flex flex-col h-full w-full overflow-hidden">
      {/* Top Global Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-2 flex items-center gap-3 shrink-0">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4 text-blue-600" />
          <h1 className="text-sm font-bold text-slate-900 whitespace-nowrap">
            Question Audit
          </h1>
        </div>

        <Select
          value={payload.dsId}
          disabled={loadingDs}
          onValueChange={(val) => {
            if (val) void switchDataset(val);
          }}
        >
          <SelectTrigger
            size="sm"
            aria-label="Select dataset CSV"
            className="bg-white font-mono text-xs text-slate-800 shadow-none"
          >
            <SelectValue>{(val: string | null) => val ?? payload.dsId}</SelectValue>
          </SelectTrigger>
          <SelectContent align="start">
            {datasets.map((ds) => (
              <SelectItem
                key={ds.id}
                value={ds.id}
                className="font-mono text-xs"
              >
                {ds.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {loadingDs && <Loader2 className="w-4 h-4 animate-spin text-blue-600" />}

        <div className="hidden md:flex items-center gap-2 text-xs text-slate-600 ml-1">
          <Badge
            variant="outline"
            className="bg-amber-50 text-amber-800 border-amber-200 font-medium"
          >
            {suspectsRemaining} suspects unreviewed
          </Badge>
          <span className="text-slate-300">|</span>
          <span>
            Reviewed{" "}
            <strong className="text-slate-900">
              {substantiveReviewed}/{substantiveTotal}
            </strong>{" "}
            substantive
          </span>
          <span className="text-slate-300">|</span>
          <span>
            Included{" "}
            <strong className="text-slate-900">{includedTotal}</strong>
          </span>
        </div>

        <div className="flex-1" />

        {unsavedTotal > 0 ? (
          <div className="flex items-center gap-1.5">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setStudyFilter("");
                setFamilyFilter([]);
                setTagFilter([]);
                setSearch("");
                setTriage("unsaved");
              }}
              title="Click to filter questions with unsaved changes"
              className="bg-orange-100 text-orange-900 border-orange-300 hover:bg-orange-200/80 font-semibold text-xs cursor-pointer"
            >
              <span className="w-2 h-2 rounded-full bg-orange-500" />
              {unsavedTotal} unsaved
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={handleSaveAllUnsaved}
              title="Save all staged edits to CSV and mark reviewed"
              className="bg-emerald-600 text-white hover:bg-emerald-700 font-semibold text-xs cursor-pointer"
            >
              <Save className="w-3.5 h-3.5" />
              Save All ({unsavedTotal})
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleDiscardAll}
              title="Discard all unsaved draft changes"
              className="bg-white text-slate-600 hover:bg-slate-50 text-xs cursor-pointer"
            >
              <RotateCcw className="w-3 h-3" />
              Discard All
            </Button>
          </div>
        ) : (
          <Badge
            variant="outline"
            className={[
              "rounded-md font-medium border-transparent",
              saveState === "saving"
                ? "text-amber-700 bg-amber-50"
                : saveState === "error"
                  ? "text-rose-700 bg-rose-50"
                  : "text-emerald-700 bg-emerald-50",
            ].join(" ")}
          >
            {saveMessage}
          </Badge>
        )}

        {unsavedTotal > 0 && saveState === "error" && (
          <Badge
            variant="outline"
            className="rounded-md font-medium text-rose-700 bg-rose-50 border-rose-200"
          >
            {saveMessage}
          </Badge>
        )}

        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => setShowBulk((v) => !v)}
          className="bg-white text-slate-700 text-xs cursor-pointer"
        >
          <Layers className="w-3.5 h-3.5" />
          Bulk ({filteredIndices.length})
        </Button>

        {payload.hasNotebook && (
          <Button
            type="button"
            size="sm"
            disabled={rebuilding}
            onClick={() => void handleRebuild()}
            title={
              unsavedTotal > 0
                ? `Save or discard ${unsavedTotal} unsaved question(s) before rebuilding Parquet`
                : "Compile saved CSV into Parquet via data.ipynb"
            }
            className={[
              "text-xs font-semibold cursor-pointer",
              unsavedTotal > 0
                ? "bg-slate-200 text-slate-600 hover:bg-slate-300"
                : "bg-blue-600 text-white hover:bg-blue-700",
            ].join(" ")}
          >
            {rebuilding ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )}
            {rebuilding ? "Rebuilding..." : "Rebuild Parquet"}
          </Button>
        )}
      </header>

      {/* Optional Rebuild / Bulk Banner */}
      {rebuildBanner && (
        <div className="bg-emerald-50 border-b border-emerald-200 px-4 py-1.5 text-xs text-emerald-900 flex items-center justify-between">
          <span className="font-mono">{rebuildBanner}</span>
          <button
            type="button"
            onClick={() => setRebuildBanner(null)}
            className="text-emerald-700 hover:underline cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {showBulk && (
        <div className="bg-slate-100 border-b border-slate-200 px-4 py-2 flex items-center gap-2 flex-wrap text-xs">
          <span className="font-semibold text-slate-700">
            Stage on {filteredIndices.length} filtered rows:
          </span>
          <Button
            type="button"
            size="xs"
            onClick={() =>
              stageRowsPatch(
                filteredIndices,
                payload.schemaKind === "question_audit"
                  ? { include: "True", kev_type: "score" }
                  : { type: "score" }
              )
            }
            className="bg-violet-600 hover:bg-violet-700 text-white cursor-pointer"
          >
            Set Score
          </Button>
          <Button
            type="button"
            size="xs"
            onClick={() =>
              stageRowsPatch(
                filteredIndices,
                payload.schemaKind === "question_audit"
                  ? { include: "True", kev_type: "choice" }
                  : { type: "choice" }
              )
            }
            className="bg-blue-600 hover:bg-blue-700 text-white cursor-pointer"
          >
            Set Choice
          </Button>
          <Button
            type="button"
            size="xs"
            onClick={() =>
              stageRowsPatch(
                filteredIndices,
                payload.schemaKind === "question_audit"
                  ? { include: "False", kev_type: "" }
                  : { type: "exclude" }
              )
            }
            className="bg-rose-600 hover:bg-rose-700 text-white cursor-pointer"
          >
            Set Exclude
          </Button>
          <Button
            type="button"
            size="xs"
            onClick={() =>
              void commitRowEntries(
                filteredIndices.map((rowIdx) => ({
                  rowIdx,
                  patch: draftPatches[rowIdx] ?? {},
                  ...(rowIdx in draftOverrides
                    ? { override: draftOverrides[rowIdx] }
                    : {}),
                  ...(rowIdx in draftTags ? { tags: draftTags[rowIdx] } : {}),
                  reviewed: true,
                }))
              )
            }
            className="bg-emerald-600 hover:bg-emerald-700 text-white cursor-pointer"
          >
            <Check className="w-3.5 h-3.5" />
            Save &amp; Mark Filtered Reviewed
          </Button>
          <div className="inline-flex items-center gap-1.5 ml-2 pl-2 border-l border-slate-300">
            <span className="text-slate-600 font-medium">Tag:</span>
            <Input
              type="text"
              value={bulkTagInput}
              onChange={(e) => setBulkTagInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  stageBulkAddTags(filteredIndices, bulkTagInput);
                }
              }}
              placeholder="e.g. revisit"
              className="h-6 w-28 bg-white font-mono text-xs px-2"
            />
            <Button
              type="button"
              variant="outline"
              size="xs"
              disabled={!bulkTagInput.trim()}
              onClick={() => stageBulkAddTags(filteredIndices, bulkTagInput)}
              className="bg-white cursor-pointer"
            >
              Add Tag
            </Button>
          </div>
          {payload.columns.includes("context_sep") && (
            <div className="inline-flex items-center gap-1.5 ml-2 pl-2 border-l border-slate-300">
              <span className="text-slate-600 font-medium">Delimiter:</span>
              <Input
                type="text"
                value={bulkSepInput}
                onChange={(e) => setBulkSepInput(e.target.value)}
                onBlur={() => {
                  if (bulkSepInput === "") setBulkSepInput("\\n\\n");
                }}
                placeholder="\\n\\n"
                className="h-6 w-24 bg-white font-mono text-xs px-2"
              />
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() => {
                  const sepVal =
                    bulkSepInput === "" || bulkSepInput === "\\n\\n"
                      ? ""
                      : bulkSepInput;
                  stageRowsPatch(filteredIndices, {
                    context_sep: sepVal,
                  });
                  setDraftResetSeq((n) => n + 1);
                }}
                className="bg-white cursor-pointer"
              >
                Apply
              </Button>
            </div>
          )}
          <span className="text-[11px] text-slate-500">
            (Bulk edits stage as Unsaved until you click Save All or Confirm)
          </span>
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => setShowBulk(false)}
            className="ml-auto text-slate-500 hover:text-slate-800 cursor-pointer"
          >
            Close
          </Button>
        </div>
      )}

      {/* Main Split Workspace */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <aside className="w-[390px] shrink-0 bg-white border-r border-slate-200 flex flex-col h-full overflow-hidden">
          {/* Triage Tabs & Filters */}
          <div className="p-2.5 border-b border-slate-200 space-y-2 bg-slate-50/60">
            <div className="flex flex-wrap gap-1">
              {(
                [
                  ["suspects", "Suspects"],
                  ["unsaved", "Unsaved"],
                  ["overridden", "Overridden"],
                  ["tagged", "Tagged"],
                  ["unreviewed", "Unreviewed"],
                  ["substantive", "Substantive"],
                  ["included", "Included"],
                  ["excluded", "Excluded"],
                  ["all", "All"],
                ] as [TriageMode, string][]
              ).map(([mode, label]) => {
                const active = triage === mode;
                const count = tabCounts[mode];
                if (
                  (mode === "overridden" || mode === "tagged") &&
                  count === 0 &&
                  !active
                ) {
                  return null;
                }
                return (
                  <Button
                    key={mode}
                    type="button"
                    variant={active ? "default" : "outline"}
                    size="xs"
                    onClick={() => setTriage(mode)}
                    className={[
                      "text-[11px] font-semibold cursor-pointer shadow-none",
                      active
                        ? mode === "suspects"
                          ? "bg-amber-500 hover:bg-amber-600 text-white border-transparent"
                          : mode === "unsaved"
                            ? "bg-orange-600 hover:bg-orange-700 text-white border-transparent"
                            : mode === "overridden"
                              ? "bg-cyan-700 hover:bg-cyan-800 text-white border-transparent"
                              : mode === "tagged"
                                ? "bg-indigo-600 hover:bg-indigo-700 text-white border-transparent"
                                : "bg-slate-900 hover:bg-slate-800 text-white border-transparent"
                        : mode === "unsaved" && count > 0
                          ? "bg-orange-50 text-orange-800 border-orange-300 hover:bg-orange-100"
                          : mode === "overridden" && count > 0
                            ? "bg-cyan-50 text-cyan-800 border-cyan-300 hover:bg-cyan-100"
                            : mode === "tagged" && count > 0
                              ? "bg-indigo-50 text-indigo-800 border-indigo-300 hover:bg-indigo-100"
                              : "bg-white text-slate-600 border-slate-200 hover:bg-slate-100",
                    ].join(" ")}
                  >
                    {label} ({count})
                  </Button>
                );
              })}
            </div>

            {/* Format-skipped visibility toggle */}
            {formatSkippedCount > 0 &&
              triage !== "all" &&
              triage !== "unsaved" && (
              <label className="flex items-center justify-between px-2 py-1 rounded-md bg-slate-100/90 border border-slate-200/80 text-[11px] text-slate-600 cursor-pointer select-none">
                <span>
                  Show format-skipped (DB / TE / Slider / Multi) dimmed
                </span>
                <span className="inline-flex items-center gap-1.5 font-mono text-slate-500">
                  ({formatSkippedCount})
                  <Checkbox
                    checked={showFormatSkipped}
                    onCheckedChange={(checked) =>
                      setShowFormatSkipped(Boolean(checked))
                    }
                    className="bg-white"
                  />
                </span>
              </label>
            )}

            {/* Dropdown & Multiselect filters */}
            <div className="grid grid-cols-2 gap-1.5">
              {studyOptions.length > 0 && (
                <Select
                  value={studyFilter || "__all__"}
                  onValueChange={(val) =>
                    setStudyFilter(!val || val === "__all__" ? "" : val)
                  }
                >
                  <SelectTrigger
                    size="sm"
                    className="w-full bg-white text-xs text-slate-700 shadow-none"
                  >
                    <SelectValue>
                      {(val: string | null) => {
                        if (!val || val === "__all__") {
                          return `All ${hasStudyCol ? "studies" : "blocks"} (${studyOptions.length})`;
                        }
                        const match = studyOptions.find(([s]) => s === val);
                        return match ? `${match[0]} (${match[1]})` : val;
                      }}
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent align="start">
                    <SelectItem value="__all__" className="text-xs">
                      All {hasStudyCol ? "studies" : "blocks"} ({studyOptions.length})
                    </SelectItem>
                    {studyOptions.map(([s, c]) => (
                      <SelectItem key={s} value={s} className="text-xs">
                        {s} ({c})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}

              <MultiSelectFilter
                options={familyOptions}
                selected={familyFilter}
                onChange={setFamilyFilter}
                allLabel={
                  payload.schemaKind === "question_audit"
                    ? "All families"
                    : "All types"
                }
                singularNoun={
                  payload.schemaKind === "question_audit" ? "family" : "type"
                }
                pluralNoun={
                  payload.schemaKind === "question_audit" ? "families" : "types"
                }
                searchPlaceholder={
                  payload.schemaKind === "question_audit"
                    ? "Search families..."
                    : "Search types..."
                }
                className={studyOptions.length === 0 ? "col-span-2" : ""}
              />

              {tagOptions.length > 0 && (
                <MultiSelectFilter
                  options={tagOptions}
                  selected={tagFilter}
                  onChange={setTagFilter}
                  allLabel="All tags"
                  singularNoun="tag"
                  pluralNoun="tags"
                  searchPlaceholder="Filter tags..."
                  className="col-span-2"
                />
              )}
            </div>

            {/* Search Input */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
              <Input
                ref={searchInputRef}
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search QID, stem, options, #tags, notes... (/)"
                className="h-7 w-full pl-8 pr-6 text-xs bg-white shadow-none"
              />
              {search && (
                <button
                  type="button"
                  aria-label="Clear search"
                  onClick={() => setSearch("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 cursor-pointer"
                >
                  <XIcon className="size-3" />
                </button>
              )}
            </div>
          </div>

          {/* Scrollable Question List */}
          <div className="flex-1 overflow-y-auto">
            {filteredRows.length === 0 ? (
              <div className="p-6 text-center text-xs text-slate-400">
                No questions match current filters.
              </div>
            ) : (
              filteredRows.map((r) => {
                const subItems = subItemsByRowKey[r.rowKey] ?? [];
                const qLower = search.trim().toLowerCase();
                const searchMatchesSub =
                  Boolean(qLower) &&
                  subItems.some(
                    (s) =>
                      s.label.toLowerCase().includes(qLower) ||
                      s.id.toLowerCase().includes(qLower)
                  );
                const isRowSelected = r.rowIdx === selectedRowIdx;
                const isExpanded =
                  subItems.length > 1 &&
                  (searchMatchesSub ||
                    (expandedGroups[r.rowKey] ?? isRowSelected));
                return (
                  <SidebarRow
                    key={r.rowKey}
                    row={r}
                    schemaKind={payload.schemaKind}
                    subItems={subItems}
                    overrideStem={effectiveOverridesByRowKey[r.rowKey]?.stem}
                    isSelected={isRowSelected}
                    isExpanded={isExpanded}
                    activeSubItemIdx={isRowSelected ? selectedSubItemIdx : -1}
                    onSelect={handleSelectRow}
                    onToggleExpand={handleToggleGroupExpand}
                    onSelectSubItem={handleSelectSubItem}
                  />
                );
              })
            )}
          </div>
        </aside>

        {/* Right Detail Pane */}
        <DetailPane
          key={`${currentRow?.rowKey ?? "empty"}:${draftResetSeq}`}
          dsId={payload.dsId}
          row={currentRow}
          savedData={currentSavedRow?.data}
          savedOverride={currentSavedOverride}
          effectiveOverride={currentEffectiveOverride}
          savedTags={currentSavedTags}
          allTagCounts={tagOptions}
          detail={currentDetail}
          studyCandidates={currentStudyCandidates}
          schemaKind={payload.schemaKind}
          columns={payload.columns}
          positionIndex={posInFiltered}
          totalFiltered={filteredRows.length}
          activeSubItemIdx={selectedSubItemIdx}
          onSelectSubItemIdx={setSelectedSubItemIdx}
          onPrev={goPrev}
          onNext={goNext}
          onStageRow={handleStageSingleRow}
          onStageOverride={handleStageOverride}
          onStageTags={handleStageTags}
          onFilterByTag={handleFilterByTag}
          onSaveRow={handleSaveSingleRow}
          onDiscardRow={handleDiscardRow}
          onToggleReviewed={handleToggleReviewed}
          onConfirmAndNext={handleConfirmAndNext}
        />
      </div>
    </div>
  );
}
