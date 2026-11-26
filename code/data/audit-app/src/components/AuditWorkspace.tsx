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
  ListChecks,
  Loader2,
  Play,
  RotateCcw,
  Save,
  Search,
  Users,
  XIcon,
  BookOpen,
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
import {
  hasCodebookDiff,
  type SourceCheckPayload,
} from "@/lib/sourceCheckTypes";
import type {
  AuditFlag,
  AuditRow,
  DatasetInfo,
  DatasetPayload,
  DecisionFilter,
  InitialViewState,
  QuestionOverride,
  ReviewFilter,
  SidebarSubItem,
  WorkspaceTab,
} from "@/lib/types";
import { DetailPane } from "./DetailPane";
import { MultiSelectFilter } from "./MultiSelectFilter";
import { PersonasPane } from "./PersonasPane";
import { SourcePane } from "./SourcePane";
import { SidebarRow } from "./SidebarRow";

const LAST_DS_STORAGE_KEY = "audit-ui-last-ds:v1";

function dsStorageKey(dsId: string): string {
  return `audit-ui-ds:v1:${dsId}`;
}

const AUDIT_FLAG_ORDER: AuditFlag[] = [
  "blocking",
  "suspect",
  "unsaved",
  "overridden",
  "tagged",
  "option_relabeled",
];

const CODEBOOK_FLAG_ORDER: AuditFlag[] = [
  "codebook_diff",
  "text_changed",
  "options_differ",
  "not_in_source",
  "identical",
];

const ALL_FLAG_ORDER: AuditFlag[] = [...AUDIT_FLAG_ORDER, ...CODEBOOK_FLAG_ORDER];

const VALID_AUDIT_FLAGS = new Set<AuditFlag>(ALL_FLAG_ORDER);

const FLAG_LABELS: Record<AuditFlag, string> = {
  blocking: "Blocks export",
  suspect: "Suspect",
  unsaved: "Unsaved",
  overridden: "Overridden",
  tagged: "Tagged",
  option_relabeled: "Relabeled",
  codebook_diff: "Codebook diff",
  text_changed: "Text changed",
  options_differ: "Options differ",
  not_in_source: "Not in codebook",
  identical: "Identical",
};

const UNCATEGORIZED_CATEGORY = "__uncategorized__";

interface StoredDatasetView {
  rowKey?: string;
  reviewFilter?: ReviewFilter;
  decisionFilter?: DecisionFilter;
  flags?: AuditFlag[];
  flagsExplicit?: boolean;
  // Legacy fields from before review/tagging unification (migrated on load)
  triage?: string;
  sourceFilter?: string;
  sourceReviewFilter?: string;
  showFormatSkipped?: boolean;
  studyFilter?: string;
  familyFilter?: string[];
  categoryFilter?: string[];
  tagFilter?: string[];
  search?: string;
  draftsByRowKey?: Record<string, Record<string, string>>;
  draftOverridesByRowKey?: Record<string, QuestionOverride | null>;
  draftTagsByRowKey?: Record<string, string[]>;
}

function resolveStoredFilters(
  stored: StoredDatasetView | null,
  hasAnyDrafts: boolean
): {
  reviewFilter: ReviewFilter;
  decisionFilter: DecisionFilter;
  flags: AuditFlag[];
  showFormatSkipped: boolean;
} {
  if (!stored) {
    return {
      reviewFilter: "all",
      decisionFilter: "all",
      flags: [],
      showFormatSkipped: false,
    };
  }

  const hasNewFormat =
    stored.reviewFilter !== undefined ||
    stored.decisionFilter !== undefined ||
    stored.flags !== undefined;

  if (hasNewFormat) {
    const rev: ReviewFilter =
      stored.reviewFilter === "unreviewed" || stored.reviewFilter === "reviewed"
        ? stored.reviewFilter
        : "all";
    const dec: DecisionFilter =
      stored.decisionFilter === "included" || stored.decisionFilter === "excluded"
        ? stored.decisionFilter
        : "all";
    // Ignore auto-saved suspect flags written before default-suspect removal
    let flags = stored.flagsExplicit
      ? (stored.flags ?? []).filter((f): f is AuditFlag =>
          VALID_AUDIT_FLAGS.has(f)
        )
      : (stored.flags ?? []).filter(
          (f): f is AuditFlag => VALID_AUDIT_FLAGS.has(f) && f !== "suspect"
        );
    if (!hasAnyDrafts && flags.includes("unsaved")) {
      flags = flags.filter((f) => f !== "unsaved");
    }
    return {
      reviewFilter: rev,
      decisionFilter: dec,
      flags,
      showFormatSkipped: stored.showFormatSkipped ?? false,
    };
  }

  // Migrate legacy stored fields
  let rev: ReviewFilter = "all";
  let dec: DecisionFilter = "all";
  let skip = stored.showFormatSkipped ?? false;
  const flagSet = new Set<AuditFlag>();

  if (stored.triage === "blocking") flagSet.add("blocking");
  else if (stored.triage === "suspects") flagSet.add("suspect");
  else if (stored.triage === "unsaved" && hasAnyDrafts) flagSet.add("unsaved");
  else if (stored.triage === "overridden") flagSet.add("overridden");
  else if (stored.triage === "tagged") flagSet.add("tagged");
  else if (stored.triage === "unreviewed") rev = "unreviewed";
  else if (stored.triage === "included") dec = "included";
  else if (stored.triage === "excluded") dec = "excluded";
  else if (stored.triage === "all") skip = true;

  if (stored.sourceFilter === "needs_check") flagSet.add("codebook_diff");
  else if (stored.sourceFilter === "flagged") flagSet.add("suspect");
  else if (
    stored.sourceFilter &&
    VALID_AUDIT_FLAGS.has(stored.sourceFilter as AuditFlag)
  ) {
    flagSet.add(stored.sourceFilter as AuditFlag);
  }

  if (stored.sourceReviewFilter === "unchecked") rev = "unreviewed";
  else if (stored.sourceReviewFilter === "checked") rev = "reviewed";

  return {
    reviewFilter: rev,
    decisionFilter: dec,
    flags: Array.from(flagSet),
    showFormatSkipped: skip,
  };
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
  flags?: AuditFlag[]
): number {
  if (targetRowKey) {
    const found = dsPayload.rows.find((r) => r.rowKey === targetRowKey);
    if (found) return found.rowIdx;
  }
  if (flags?.includes("suspect")) {
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
  const [activeTab, setActiveTab] = useState<WorkspaceTab>(
    initialView.tab ?? "questions"
  );
  const [detailTab, setDetailTab] = useState<"audit" | "marginals">("audit");
  const [marginalPersona, setMarginalPersona] = useState<string>("");
  const [personasRefreshSeq, setPersonasRefreshSeq] = useState(0);
  const [sourceCheck, setSourceCheck] = useState<SourceCheckPayload | null>(null);
  const [sourceCheckLoading, setSourceCheckLoading] = useState(true);
  const [sourceCheckError, setSourceCheckError] = useState<string | null>(null);
  const sourceReqSeqRef = useRef(0);
  const [loadingDs, setLoadingDs] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  const [reviewFilter, setReviewFilterState] = useState<ReviewFilter>(
    initialView.reviewFilter ?? "all"
  );
  const [decisionFilter, setDecisionFilterState] = useState<DecisionFilter>(
    initialView.decisionFilter ?? "all"
  );
  const [activeFlags, setActiveFlagsState] = useState<AuditFlag[]>(
    () => initialView.flags ?? []
  );
  const [showFormatSkipped, setShowFormatSkippedState] = useState<boolean>(
    initialView.skip ?? false
  );
  const [studyFilter, setStudyFilterState] = useState(initialView.study ?? "");
  const [familyFilter, setFamilyFilterState] = useState<string[]>(
    initialView.families ?? []
  );
  const [categoryFilter, setCategoryFilterState] = useState<string[]>(
    (initialView.categories ?? []).slice(0, 1)
  );
  const [tagFilter, setTagFilterState] = useState<string[]>(
    initialView.tags ?? []
  );
  const [search, setSearchState] = useState(initialView.q ?? "");
  const [bulkSepInput, setBulkSepInput] = useState("\\n\\n");
  const [bulkTagInput, setBulkTagInput] = useState("");
  const [bulkCategoryInput, setBulkCategoryInput] = useState("");

  const [selectedRowIdx, setSelectedRowIdx] = useState<number>(() =>
    resolveInitialRowIdx(initialPayload, initialView.rowKey, initialView.flags)
  );
  // Keep a restored or just-edited/just-toggled row open even if the active filter hides it;
  // any manual filter/search change or explicit row selection clears this.
  const [keepSelectedRow, setKeepSelectedRow] = useState<boolean>(
    Boolean(initialView.rowKey)
  );
  const [selectedSubItemIdx, setSelectedSubItemIdx] = useState(0);
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>(
    {}
  );

  const setReviewFilter = useCallback((next: ReviewFilter) => {
    setKeepSelectedRow(false);
    setReviewFilterState(next);
  }, []);
  const setDecisionFilter = useCallback((next: DecisionFilter) => {
    setKeepSelectedRow(false);
    setDecisionFilterState(next);
  }, []);
  const setActiveFlags = useCallback(
    (next: AuditFlag[] | ((prev: AuditFlag[]) => AuditFlag[])) => {
      setKeepSelectedRow(false);
      setActiveFlagsState(next);
    },
    []
  );
  const toggleFlag = useCallback((flag: AuditFlag) => {
    setKeepSelectedRow(false);
    setActiveFlagsState((prev) =>
      prev.includes(flag) ? prev.filter((f) => f !== flag) : [...prev, flag]
    );
  }, []);
  const resetUnifiedFilters = useCallback(() => {
    setKeepSelectedRow(false);
    setReviewFilterState("all");
    setDecisionFilterState("all");
    setActiveFlagsState([]);
    setCategoryFilterState([]);
  }, []);
  const toggleCategory = useCallback((cat: string) => {
    setKeepSelectedRow(false);
    setCategoryFilterState((prev) =>
      prev.length === 1 && prev[0] === cat ? [] : [cat]
    );
  }, []);
  const clearCategories = useCallback(() => {
    setKeepSelectedRow(false);
    setCategoryFilterState([]);
  }, []);
  const focusFlag = useCallback((flag: AuditFlag) => {
    setKeepSelectedRow(false);
    setReviewFilterState("all");
    setDecisionFilterState("all");
    setActiveFlagsState([flag]);
  }, []);
  const setShowFormatSkipped = useCallback((next: boolean) => {
    setKeepSelectedRow(false);
    setShowFormatSkippedState(next);
  }, []);
  const setStudyFilter = useCallback((next: string) => {
    setKeepSelectedRow(false);
    setStudyFilterState(next);
  }, []);
  const setFamilyFilter = useCallback(
    (next: string[] | ((prev: string[]) => string[])) => {
      setKeepSelectedRow(false);
      setFamilyFilterState(next);
    },
    []
  );
  const setTagFilter = useCallback(
    (next: string[] | ((prev: string[]) => string[])) => {
      setKeepSelectedRow(false);
      setTagFilterState(next);
    },
    []
  );
  const setSearch = useCallback((next: string) => {
    setKeepSelectedRow(false);
    setSearchState(next);
  }, []);

  const fetchSourceCheck = useCallback(
    async (
      dsId: string,
      opts?: { clear?: boolean; keepCurrent?: boolean }
    ): Promise<SourceCheckPayload | null> => {
      const seq = ++sourceReqSeqRef.current;
      if (opts?.clear) {
        setSourceCheck(null);
      }
      setSourceCheckLoading(true);
      setSourceCheckError(null);
      try {
        const res = await fetch(`/api/source-check?ds=${encodeURIComponent(dsId)}`, {
          cache: "no-store",
        });
        const body = (await res.json()) as SourceCheckPayload & { error?: string };
        if (!res.ok || body.error) {
          throw new Error(body.error ?? `HTTP ${res.status}`);
        }
        if (seq !== sourceReqSeqRef.current) return null;
        if (opts?.keepCurrent) {
          setKeepSelectedRow(true);
        }
        setSourceCheck(body);
        return body;
      } catch (err: unknown) {
        if (seq !== sourceReqSeqRef.current) return null;
        setSourceCheckError(err instanceof Error ? err.message : String(err));
        return null;
      } finally {
        if (seq === sourceReqSeqRef.current) {
          setSourceCheckLoading(false);
        }
      }
    },
    []
  );

  useEffect(() => {
    void fetchSourceCheck(payload.dsId, { clear: true });
  }, [payload.dsId, fetchSourceCheck]);

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
  const [rebuildingAll, setRebuildingAll] = useState(false);
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
        const restoredFilters = resolveStoredFilters(stored, hasAnyDrafts);
        const nextFlags =
          nextDsId !== payload.dsId ? [] : restoredFilters.flags;

        setPayload(nextPayload);
        if (nextDsId !== payload.dsId) {
          setMarginalPersona("");
        }
        setDraftPatches(restoredDrafts);
        setDraftOverrides(restoredOverrides);
        setDraftTags(restoredTags);
        setDraftResetSeq((n) => n + 1);
        setShowFormatSkippedState(restoredFilters.showFormatSkipped);
        setStudyFilterState(stored?.studyFilter ?? "");
        setFamilyFilterState(stored?.familyFilter ?? []);
        setCategoryFilterState((stored?.categoryFilter ?? []).slice(0, 1));
        setTagFilterState(stored?.tagFilter ?? []);
        setSearchState(stored?.search ?? "");
        setReviewFilterState(restoredFilters.reviewFilter);
        setDecisionFilterState(restoredFilters.decisionFilter);
        setActiveFlagsState(nextFlags);
        setKeepSelectedRow(Boolean(stored?.rowKey));
        setSelectedRowIdx(
          resolveInitialRowIdx(nextPayload, stored?.rowKey, nextFlags)
        );
        setSelectedSubItemIdx(0);
        setExpandedGroups({});
        setSaveState("saved");
        setSaveMessage("All saved");
        if (nextDsId === payload.dsId) {
          void fetchSourceCheck(nextDsId, { keepCurrent: true });
        }
      } catch (err: unknown) {
        setSaveState("error");
        setSaveMessage(err instanceof Error ? err.message : String(err));
      } finally {
        setLoadingDs(false);
      }
    },
    [payload.dsId, draftPatches, draftOverrides, draftTags, fetchSourceCheck]
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
            initialView.flags
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
      const restoredFilters = resolveStoredFilters(stored, hasAnyDrafts);
      if (hasAnyDrafts) {
        setDraftPatches(restoredDrafts);
        setDraftOverrides(restoredOverrides);
        setDraftTags(restoredTags);
        setDraftResetSeq((n) => n + 1);
      }
      setShowFormatSkippedState(restoredFilters.showFormatSkipped);
      setStudyFilterState(stored.studyFilter ?? "");
      setFamilyFilterState(stored.familyFilter ?? []);
      setCategoryFilterState((stored.categoryFilter ?? []).slice(0, 1));
      setTagFilterState(stored.tagFilter ?? []);
      setSearchState(stored.search ?? "");
      setReviewFilterState(restoredFilters.reviewFilter);
      setDecisionFilterState(restoredFilters.decisionFilter);
      setActiveFlagsState(restoredFilters.flags);
      setKeepSelectedRow(Boolean(stored.rowKey));
      setSelectedRowIdx(
        resolveInitialRowIdx(initialPayload, stored.rowKey, restoredFilters.flags)
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

  const sourceConfigured = Boolean(sourceCheck?.configured);

  // Rows matching search + dropdowns (before triage, format-skip & source filters)
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
        const srcItem = sourceConfigured ? sourceCheck?.items[r.rowKey] : undefined;
        const srcText = srcItem
          ? `${srcItem.source_text ?? ""} ${srcItem.warnings.join(" ")}`
          : "";
        const hay = `${Object.values(d).join(" ")} ${subText} ${ovText} ${tagText} ${srcText}`.toLowerCase();
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
    sourceConfigured,
    sourceCheck,
    studyFilter,
    familyFilter,
    tagFilter,
    search,
  ]);

  const matchesSkipVisibility = useCallback(
    (r: AuditRow, flags: AuditFlag[], includeSkipped: boolean): boolean => {
      if (includeSkipped || !r.isFormatSkipped) return true;
      if (isRowIncluded(payload.schemaKind, r.data)) return true;
      if (flags.includes("unsaved") && r.isDirty) return true;
      if (flags.includes("blocking") && r.exportProblem !== "") return true;
      if (flags.includes("overridden") && r.hasOverride) return true;
      if (flags.includes("tagged") && (r.tags?.length ?? 0) > 0) return true;
      return false;
    },
    [payload.schemaKind]
  );

  const matchesReview = useCallback(
    (r: AuditRow, rev: ReviewFilter): boolean => {
      if (rev === "all") return true;
      if (rev === "unreviewed") return !r.reviewed;
      return r.reviewed;
    },
    []
  );

  const matchesDecision = useCallback(
    (r: AuditRow, dec: DecisionFilter): boolean => {
      if (dec === "all") return true;
      const inc = isRowIncluded(payload.schemaKind, r.data);
      return dec === "included" ? inc : !inc;
    },
    [payload.schemaKind]
  );

  const matchesFlag = useCallback(
    (r: AuditRow, flag: AuditFlag): boolean => {
      switch (flag) {
        case "blocking":
          return r.exportProblem !== "";
        case "suspect":
          return r.isSuspect;
        case "unsaved":
          return Boolean(r.isDirty);
        case "overridden":
          return Boolean(r.hasOverride);
        case "tagged":
          return (r.tags?.length ?? 0) > 0;
        case "option_relabeled":
          return (
            isRowIncluded(payload.schemaKind, r.data) &&
            Boolean(payload.items[r.rowKey]?.option_relabeled)
          );
        case "codebook_diff": {
          if (!sourceConfigured || !sourceCheck) return true;
          const item = sourceCheck.items[r.rowKey];
          return Boolean(item && hasCodebookDiff(item));
        }
        case "text_changed": {
          if (!sourceConfigured || !sourceCheck) return true;
          const item = sourceCheck.items[r.rowKey];
          return Boolean(item && item.text_status === "changed");
        }
        case "options_differ": {
          if (!sourceConfigured || !sourceCheck) return true;
          const item = sourceCheck.items[r.rowKey];
          return Boolean(item && item.options_status === "differ");
        }
        case "not_in_source": {
          if (!sourceConfigured || !sourceCheck) return true;
          const item = sourceCheck.items[r.rowKey];
          return Boolean(
            item && (item.text_status === "not_in_source" || item.error !== null)
          );
        }
        case "identical": {
          if (!sourceConfigured || !sourceCheck) return true;
          const item = sourceCheck.items[r.rowKey];
          return Boolean(item && !hasCodebookDiff(item));
        }
      }
    },
    [payload.schemaKind, payload.items, sourceConfigured, sourceCheck]
  );

  const matchesAllFlags = useCallback(
    (r: AuditRow, flags: AuditFlag[]): boolean =>
      flags.every((f) => matchesFlag(r, f)),
    [matchesFlag]
  );

  const hasCategoryCol = payload.columns.includes("category");

  const matchesCategories = useCallback(
    (r: AuditRow, cats: string[]): boolean => {
      if (!hasCategoryCol || cats.length === 0) return true;
      const val = (r.data.category || "").trim();
      return val
        ? cats.includes(val)
        : cats.includes(UNCATEGORIZED_CATEGORY);
    },
    [hasCategoryCol]
  );

  const formatSkippedCount = useMemo(
    () => preTriageRows.filter((r) => r.isFormatSkipped).length,
    [preTriageRows]
  );

  const visibleFlags = useMemo(
    () => (sourceConfigured ? ALL_FLAG_ORDER : AUDIT_FLAG_ORDER),
    [sourceConfigured]
  );

  const filteredRows = useMemo(
    () =>
      preTriageRows.filter(
        (r) =>
          matchesSkipVisibility(r, activeFlags, showFormatSkipped) &&
          matchesReview(r, reviewFilter) &&
          matchesDecision(r, decisionFilter) &&
          matchesAllFlags(r, activeFlags) &&
          matchesCategories(r, categoryFilter)
      ),
    [
      preTriageRows,
      matchesSkipVisibility,
      matchesReview,
      matchesDecision,
      matchesAllFlags,
      matchesCategories,
      activeFlags,
      showFormatSkipped,
      reviewFilter,
      decisionFilter,
      categoryFilter,
    ]
  );

  const reviewCounts = useMemo(() => {
    const base = preTriageRows.filter(
      (r) =>
        matchesSkipVisibility(r, activeFlags, showFormatSkipped) &&
        matchesDecision(r, decisionFilter) &&
        matchesAllFlags(r, activeFlags) &&
        matchesCategories(r, categoryFilter)
    );
    let reviewed = 0;
    for (const r of base) {
      if (r.reviewed) reviewed++;
    }
    return {
      all: base.length,
      unreviewed: base.length - reviewed,
      reviewed,
    };
  }, [
    preTriageRows,
    matchesSkipVisibility,
    matchesDecision,
    matchesAllFlags,
    matchesCategories,
    activeFlags,
    showFormatSkipped,
    decisionFilter,
    categoryFilter,
  ]);

  const decisionCounts = useMemo(() => {
    const base = preTriageRows.filter(
      (r) =>
        matchesSkipVisibility(r, activeFlags, showFormatSkipped) &&
        matchesReview(r, reviewFilter) &&
        matchesAllFlags(r, activeFlags) &&
        matchesCategories(r, categoryFilter)
    );
    let included = 0;
    for (const r of base) {
      if (isRowIncluded(payload.schemaKind, r.data)) included++;
    }
    return {
      all: base.length,
      included,
      excluded: base.length - included,
    };
  }, [
    preTriageRows,
    matchesSkipVisibility,
    matchesReview,
    matchesAllFlags,
    matchesCategories,
    activeFlags,
    showFormatSkipped,
    reviewFilter,
    categoryFilter,
    payload.schemaKind,
  ]);

  const flagCounts = useMemo(() => {
    const out = {} as Record<AuditFlag, number>;
    for (const f of ALL_FLAG_ORDER) {
      const nextFlags = activeFlags.includes(f)
        ? activeFlags
        : [...activeFlags, f];
      out[f] = preTriageRows.filter(
        (r) =>
          matchesSkipVisibility(r, nextFlags, showFormatSkipped) &&
          matchesReview(r, reviewFilter) &&
          matchesDecision(r, decisionFilter) &&
          matchesAllFlags(r, nextFlags) &&
          matchesCategories(r, categoryFilter)
      ).length;
    }
    return out;
  }, [
    preTriageRows,
    matchesSkipVisibility,
    matchesReview,
    matchesDecision,
    matchesAllFlags,
    matchesCategories,
    activeFlags,
    showFormatSkipped,
    reviewFilter,
    decisionFilter,
    categoryFilter,
  ]);

  // All distinct categories across the dataset (stable alphabetical order so chips never jump)
  const categoryOptions = useMemo(() => {
    if (!hasCategoryCol) return [];
    const counts = new Map<string, number>();
    let uncategorizedCount = 0;
    for (const r of displayRows) {
      const cat = (r.data.category || "").trim();
      if (cat) {
        counts.set(cat, (counts.get(cat) ?? 0) + 1);
      } else {
        uncategorizedCount++;
      }
    }
    for (const sel of categoryFilter) {
      if (sel !== UNCATEGORIZED_CATEGORY && !counts.has(sel)) {
        counts.set(sel, 0);
      }
    }
    const sorted = Array.from(counts.entries()).sort((a, b) =>
      a[0].localeCompare(b[0])
    );
    if (
      (uncategorizedCount > 0 && sorted.length > 0) ||
      categoryFilter.includes(UNCATEGORIZED_CATEGORY)
    ) {
      sorted.push([UNCATEGORIZED_CATEGORY, uncategorizedCount]);
    }
    return sorted;
  }, [hasCategoryCol, displayRows, categoryFilter]);

  const namedCategoryCounts = useMemo(
    () => categoryOptions.filter(([cat]) => cat !== UNCATEGORIZED_CATEGORY),
    [categoryOptions]
  );

  // Per-category counts under all other active filters (excluding categoryFilter itself so switching categories stays informative)
  const categoryCounts = useMemo(() => {
    const out: Record<string, number> = {};
    if (!hasCategoryCol) return out;
    const base = preTriageRows.filter(
      (r) =>
        matchesSkipVisibility(r, activeFlags, showFormatSkipped) &&
        matchesReview(r, reviewFilter) &&
        matchesDecision(r, decisionFilter) &&
        matchesAllFlags(r, activeFlags)
    );
    for (const r of base) {
      const key = (r.data.category || "").trim() || UNCATEGORIZED_CATEGORY;
      out[key] = (out[key] ?? 0) + 1;
    }
    return out;
  }, [
    hasCategoryCol,
    preTriageRows,
    matchesSkipVisibility,
    matchesReview,
    matchesDecision,
    matchesAllFlags,
    activeFlags,
    showFormatSkipped,
    reviewFilter,
    decisionFilter,
  ]);

  // Dataset-level counts (ignoring active flags) to decide whether optional audit flags should appear
  const flagAnyCount = useMemo(() => {
    const out = {} as Record<AuditFlag, number>;
    for (const f of AUDIT_FLAG_ORDER) {
      out[f] = preTriageRows.filter((r) => matchesFlag(r, f)).length;
    }
    return out;
  }, [preTriageRows, matchesFlag]);

  const hasActiveUnifiedFilter =
    reviewFilter !== "all" ||
    decisionFilter !== "all" ||
    activeFlags.length > 0 ||
    categoryFilter.length > 0;

  const filteredIndices = useMemo(
    () => filteredRows.map((r) => r.rowIdx),
    [filteredRows]
  );

  // Keep selection valid when filters change (only after initial restoration)
  useEffect(() => {
    if (!hydrated) return;
    if (
      keepSelectedRow &&
      selectedRowIdx >= 0 &&
      selectedRowIdx < displayRows.length
    ) {
      return;
    }
    if (filteredIndices.length === 0) {
      setSelectedRowIdx(-1);
      return;
    }
    if (!filteredIndices.includes(selectedRowIdx)) {
      setSelectedRowIdx(filteredIndices[0]);
    }
  }, [
    hydrated,
    filteredIndices,
    selectedRowIdx,
    keepSelectedRow,
    displayRows.length,
  ]);

  // Persist current place (URL query + localStorage including unsaved drafts)
  useEffect(() => {
    if (!hydrated) return;
    const activeRowKey =
      selectedRowIdx >= 0 ? payload.rows[selectedRowIdx]?.rowKey : undefined;

    const sp = new URLSearchParams();
    sp.set("ds", payload.dsId);
    if (activeTab === "personas") sp.set("tab", "personas");
    if (activeTab === "source") sp.set("tab", "source");
    if (activeRowKey) sp.set("row", activeRowKey);
    if (reviewFilter !== "all") sp.set("rev", reviewFilter);
    if (decisionFilter !== "all") sp.set("dec", decisionFilter);
    for (const f of activeFlags) {
      sp.append("flag", f);
    }
    if (showFormatSkipped) sp.set("skip", "1");
    if (studyFilter) sp.set("study", studyFilter);
    for (const fam of familyFilter) {
      sp.append("family", fam);
    }
    for (const cat of categoryFilter) {
      sp.append("cat", cat);
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
        reviewFilter,
        decisionFilter,
        flags: activeFlags,
        flagsExplicit: true,
        showFormatSkipped,
        studyFilter,
        familyFilter,
        categoryFilter,
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
    activeTab,
    selectedRowIdx,
    reviewFilter,
    decisionFilter,
    activeFlags,
    showFormatSkipped,
    studyFilter,
    familyFilter,
    categoryFilter,
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
    ): Promise<boolean> => {
      if (entries.length === 0) return true;
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
        if (
          sourceConfigured &&
          (wroteCsv || wroteOverrides || wroteTags)
        ) {
          void fetchSourceCheck(payload.dsId, { keepCurrent: true });
        }
        return true;
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
        return false;
      }
    },
    [payload.dsId, sourceConfigured, fetchSourceCheck]
  );

  const handleSaveSingleRow = useCallback(
    (rowIdx: number) => {
      const pendingPatch = draftPatches[rowIdx] ?? {};
      const hasOv = rowIdx in draftOverrides;
      const hasTg = rowIdx in draftTags;
      if (Object.keys(pendingPatch).length === 0 && !hasOv && !hasTg) return;
      setKeepSelectedRow(true);
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
      if (reviewFilter !== "all") {
        // Keep the row open after toggling so the user can immediately undo with Space or move with j / k.
        setKeepSelectedRow(true);
      }
      void commitRowEntries([{ rowIdx, patch: {}, reviewed: nextReviewed }]);
    },
    [commitRowEntries, reviewFilter]
  );

  const posInFiltered = filteredIndices.indexOf(selectedRowIdx);

  const { prevRowIdx, nextRowIdx } = useMemo(() => {
    if (filteredIndices.length === 0) {
      return { prevRowIdx: -1, nextRowIdx: -1 };
    }
    if (posInFiltered >= 0) {
      return {
        prevRowIdx: posInFiltered > 0 ? filteredIndices[posInFiltered - 1] : -1,
        nextRowIdx:
          posInFiltered < filteredIndices.length - 1
            ? filteredIndices[posInFiltered + 1]
            : -1,
      };
    }
    if (selectedRowIdx < 0) {
      return { prevRowIdx: -1, nextRowIdx: filteredIndices[0] };
    }
    let prev = -1;
    let next = -1;
    for (const idx of filteredIndices) {
      if (idx < selectedRowIdx) prev = idx;
      else if (idx > selectedRowIdx && next < 0) {
        next = idx;
        break;
      }
    }
    return { prevRowIdx: prev, nextRowIdx: next };
  }, [filteredIndices, posInFiltered, selectedRowIdx]);

  const handleSelectRow = useCallback((rowIdx: number) => {
    setKeepSelectedRow(false);
    setSelectedRowIdx(rowIdx);
    setSelectedSubItemIdx(0);
  }, []);

  /** Switch to the Source tab keeping the current row and unified sidebar intact. */
  const handleViewInSource = useCallback(
    (rowKey?: string) => {
      if (rowKey) {
        const row = payload.rows.find((r) => r.rowKey === rowKey);
        if (row && row.rowIdx !== selectedRowIdx) {
          setSelectedRowIdx(row.rowIdx);
          setSelectedSubItemIdx(0);
        }
      }
      setShowBulk(false);
      setActiveTab("source");
    },
    [payload.rows, selectedRowIdx]
  );

  /** Switch to the Questions tab keeping the current row and unified sidebar intact. */
  const handleEditInQuestions = useCallback(() => {
    setActiveTab("questions");
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
    setKeepSelectedRow(false);
    setSelectedRowIdx(rowIdx);
    setSelectedSubItemIdx(subIdx);
  }, []);

  const goPrev = useCallback(() => {
    if (prevRowIdx >= 0) {
      setKeepSelectedRow(false);
      setSelectedRowIdx(prevRowIdx);
      setSelectedSubItemIdx(0);
    }
  }, [prevRowIdx]);

  const goNext = useCallback(() => {
    if (nextRowIdx >= 0) {
      setKeepSelectedRow(false);
      setSelectedRowIdx(nextRowIdx);
      setSelectedSubItemIdx(0);
    }
  }, [nextRowIdx]);

  const handleConfirmAndNext = useCallback(async () => {
    if (selectedRowIdx < 0) return;
    const targetNext = nextRowIdx;
    const pendingPatch = draftPatches[selectedRowIdx] ?? {};
    const hasOv = selectedRowIdx in draftOverrides;
    const hasTg = selectedRowIdx in draftTags;
    if (targetNext < 0) {
      setKeepSelectedRow(true);
    }
    const ok = await commitRowEntries([
      {
        rowIdx: selectedRowIdx,
        patch: pendingPatch,
        ...(hasOv ? { override: draftOverrides[selectedRowIdx] } : {}),
        ...(hasTg ? { tags: draftTags[selectedRowIdx] } : {}),
        reviewed: true,
      },
    ]);
    if (!ok) return;
    if (targetNext >= 0) {
      setKeepSelectedRow(false);
      setSelectedRowIdx(targetNext);
      setSelectedSubItemIdx(0);
    }
  }, [
    selectedRowIdx,
    nextRowIdx,
    draftPatches,
    draftOverrides,
    draftTags,
    commitRowEntries,
  ]);

  // Global keyboard shortcuts: unified navigation & staging across Questions and Source.
  useEffect(() => {
    if (activeTab !== "questions" && activeTab !== "source") return;
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
      if (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        targetEl?.isContentEditable
      ) {
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
      } else if (e.key === "v" && activeTab === "questions") {
        e.preventDefault();
        const current = payload.rows[selectedRowIdx];
        if (current) handleViewInSource(current.rowKey);
      } else if (e.key === "e" && activeTab === "source") {
        e.preventDefault();
        handleEditInQuestions();
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
        void handleConfirmAndNext();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    activeTab,
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
    handleViewInSource,
    handleEditInQuestions,
  ]);

  const handleRebuild = async () => {
    if (unsavedTotal > 0) {
      setSaveState("error");
      setSaveMessage(
        `Save or discard ${unsavedTotal} unsaved question(s) before rebuilding Parquet`
      );
      focusFlag("unsaved");
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
      setPersonasRefreshSeq((n) => n + 1);
      // Reload first: switchDataset clears the banner.
      await switchDataset(payload.dsId, true);
      setRebuildBanner(`Parquet rebuilt: ${json.summary}`);
    } catch (err: unknown) {
      // The rebuild re-indexes before exporting, so reload to show fresh 'Blocks export' flags.
      // Reload first: switchDataset resets the status message, and the error must stay visible.
      await switchDataset(payload.dsId, true);
      setSaveState("error");
      setSaveMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setRebuilding(false);
    }
  };

  const handleRebuildAll = async () => {
    if (unsavedTotal > 0) {
      setSaveState("error");
      setSaveMessage(
        `Save or discard ${unsavedTotal} unsaved question(s) before rebuilding all Parquets`
      );
      focusFlag("unsaved");
      return;
    }
    setRebuildingAll(true);
    setRebuildBanner(null);
    try {
      const res = await fetch("/api/rebuild-all", { method: "POST" });
      const json = await res.json();
      if (json.error) throw new Error(json.error);
      const results = json.results as { dsId: string; ok: boolean; summary: string; stderr: string }[];
      const done = results.filter((r) => r.ok).map((r) => `${r.dsId.split("/")[0]}: ${r.summary}`);
      if (!json.ok) {
        const failed = results[results.length - 1];
        const skipped = (json.skipped as string[]).map((d) => d.split("/")[0]);
        throw new Error(
          `Rebuilt ${done.length}; FAILED ${failed.dsId.split("/")[0]}` +
            (skipped.length ? `; not run: ${skipped.join(", ")}` : "") +
            `\n${failed.stderr}`
        );
      }
      setPersonasRefreshSeq((n) => n + 1);
      await switchDataset(payload.dsId, true);
      setRebuildBanner(`All ${done.length} Parquets rebuilt. ${done.join(" || ")}`);
    } catch (err: unknown) {
      await switchDataset(payload.dsId, true);
      setSaveState("error");
      setSaveMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setRebuildingAll(false);
    }
  };

  // Header stats
  const substantiveTotal = displayRows.filter((r) => !r.isFormatSkipped).length;
  const substantiveReviewed = displayRows.filter(
    (r) => !r.isFormatSkipped && r.reviewed
  ).length;
  const exportBlockingTotal = displayRows.filter(
    (r) => r.exportProblem !== ""
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

  const handleFilterByCategory = useCallback((cat: string) => {
    setKeepSelectedRow(false);
    setCategoryFilterState([cat]);
  }, []);

  return (
    <div className="flex flex-col h-full w-full overflow-hidden">
      {/* Top Global Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-2 flex flex-wrap items-center gap-x-3 gap-y-2 shrink-0">
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

        {/* Top-level Workspace View Tabs: Questions vs Personas */}
        <div
          role="tablist"
          aria-label="Workspace view"
          className="inline-flex items-center rounded-lg border border-slate-200 bg-slate-100/80 p-0.5 gap-0.5"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "questions"}
            onClick={() => setActiveTab("questions")}
            className={[
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-colors cursor-pointer",
              activeTab === "questions"
                ? "bg-white text-slate-900 shadow-2xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            <ListChecks className="w-3.5 h-3.5" />
            Questions
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "personas"}
            onClick={() => {
              setActiveTab("personas");
              setShowBulk(false);
            }}
            className={[
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-colors cursor-pointer",
              activeTab === "personas"
                ? "bg-white text-blue-700 shadow-2xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            <Users className="w-3.5 h-3.5" />
            Personas
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "source"}
            onClick={() => {
              setActiveTab("source");
              setShowBulk(false);
            }}
            title="Compare each question's wording and options with the dataset's codebook"
            className={[
              "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold transition-colors cursor-pointer",
              activeTab === "source"
                ? "bg-white text-emerald-700 shadow-2xs"
                : "text-slate-600 hover:text-slate-900",
            ].join(" ")}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Source
          </button>
        </div>

        {activeTab !== "personas" && (
        <div className="hidden xl:flex items-center gap-2 text-xs text-slate-600 ml-1 whitespace-nowrap">
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
        )}

        <div className="flex-1" />

        <div className="flex flex-wrap items-center justify-end gap-2 ml-auto min-w-0 max-w-full">
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
                focusFlag("unsaved");
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
              "max-w-[18rem] truncate block",
            ].join(" ")}
            title={saveMessage}
          >
            {saveMessage}
          </Badge>
        )}

        {unsavedTotal > 0 && saveState === "error" && (
          <Badge
            variant="outline"
            className="rounded-md font-medium text-rose-700 bg-rose-50 border-rose-200 max-w-[18rem] truncate block"
            title={saveMessage}
          >
            {saveMessage}
          </Badge>
        )}

        {activeTab === "questions" && (
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
        )}

        <div className="inline-flex items-center rounded-md border border-slate-200 bg-slate-50 p-0.5 gap-0.5">
          <span className="px-1.5 text-[11px] font-semibold text-slate-500 select-none">Parquet</span>
          {payload.hasNotebook && (
            <Button
              type="button"
              size="sm"
              disabled={rebuilding || rebuildingAll}
              onClick={() => void handleRebuild()}
              title={
                unsavedTotal > 0
                  ? `Save or discard ${unsavedTotal} unsaved question(s) before rebuilding Parquet`
                  : "Rebuild this dataset's Parquet from its saved CSV"
              }
              className={[
                "h-7 text-xs font-semibold cursor-pointer",
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
              {rebuilding ? "Rebuilding..." : "This dataset"}
            </Button>
          )}
          <Button
            type="button"
            size="sm"
            disabled={rebuilding || rebuildingAll}
            onClick={() => void handleRebuildAll()}
            title="Rebuild every dataset's Parquet (Twin-2K-500 first, then the rest); stops at the first failure"
            className="h-7 text-xs font-semibold cursor-pointer bg-indigo-600 text-white hover:bg-indigo-700"
          >
            {rebuildingAll ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5" />
            )}
            {rebuildingAll ? "Rebuilding all..." : "All datasets"}
          </Button>
        </div>
        </div>
      </header>

      {/* Export-blocking banner: included rows the Parquet export cannot handle (computed, not saved) */}
      {exportBlockingTotal > 0 && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-1.5 text-xs text-red-900 flex items-center justify-between gap-3">
          <span>
            <strong>{exportBlockingTotal}</strong> included question(s) cannot be exported, so the
            Parquet rebuild will fail. Exclude them first.
          </span>
          <button
            type="button"
            onClick={() => {
              focusFlag("blocking");
            }}
            className="text-red-700 font-semibold hover:underline cursor-pointer shrink-0"
          >
            Show them
          </button>
        </div>
      )}

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

      {activeTab === "questions" && showBulk && (
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
          {hasCategoryCol && (
            <div className="inline-flex items-center gap-1.5 ml-2 pl-2 border-l border-slate-300">
              <span className="text-slate-600 font-medium">Category:</span>
              <Input
                type="text"
                list="audit-category-options"
                value={bulkCategoryInput}
                onChange={(e) => setBulkCategoryInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && bulkCategoryInput.trim()) {
                    e.preventDefault();
                    e.stopPropagation();
                    stageRowsPatch(filteredIndices, {
                      category: bulkCategoryInput.trim(),
                    });
                    setBulkCategoryInput("");
                    setDraftResetSeq((n) => n + 1);
                  }
                }}
                placeholder="e.g. Work-Life Balance"
                className="h-6 w-40 bg-white text-xs px-2"
              />
              <datalist id="audit-category-options">
                {namedCategoryCounts.map(([cat]) => (
                  <option key={cat} value={cat} />
                ))}
              </datalist>
              <Button
                type="button"
                variant="outline"
                size="xs"
                disabled={!bulkCategoryInput.trim()}
                onClick={() => {
                  stageRowsPatch(filteredIndices, {
                    category: bulkCategoryInput.trim(),
                  });
                  setBulkCategoryInput("");
                  setDraftResetSeq((n) => n + 1);
                }}
                className="bg-white cursor-pointer"
              >
                Set Category
              </Button>
            </div>
          )}
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

      {/* Main Workspace */}
      {activeTab === "personas" ? (
        <PersonasPane dsId={payload.dsId} refreshSeq={personasRefreshSeq} />
      ) : (
      <div className="flex-1 flex overflow-hidden">
        {/* Unified Left Sidebar (shared between Questions and Source tabs) */}
        <aside className="w-[360px] shrink-0 bg-white border-r border-slate-200 flex flex-col h-full overflow-hidden">
          {/* Unified Filter Panel: Review + Decision + Flags & Tags */}
          <div className="p-2 border-b border-slate-200 space-y-1.5 bg-slate-50/60">
            {/* Review Status Segmented Bar (shared across Questions & Source) */}
            <div
              className="grid grid-cols-3 rounded-md border border-slate-200 bg-white p-0.5 gap-0.5"
              role="group"
              aria-label="Review status filter"
            >
              {(
                [
                  ["all", "All"],
                  ["unreviewed", "Unreviewed"],
                  ["reviewed", "Reviewed"],
                ] as [ReviewFilter, string][]
              ).map(([rf, label]) => (
                <button
                  key={rf}
                  type="button"
                  onClick={() => setReviewFilter(rf)}
                  className={[
                    "px-1.5 py-0.5 rounded text-[11px] font-semibold cursor-pointer text-center transition-colors",
                    reviewFilter === rf
                      ? rf === "reviewed"
                        ? "bg-emerald-700 text-white"
                        : "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100",
                  ].join(" ")}
                >
                  {label} ({reviewCounts[rf]})
                </button>
              ))}
            </div>

            {/* Decision Status Segmented Bar */}
            <div
              className="grid grid-cols-3 rounded-md border border-slate-200 bg-white p-0.5 gap-0.5"
              role="group"
              aria-label="Decision filter"
            >
              {(
                [
                  ["all", "All"],
                  ["included", "Included"],
                  ["excluded", "Excluded"],
                ] as [DecisionFilter, string][]
              ).map(([df, label]) => (
                <button
                  key={df}
                  type="button"
                  onClick={() => setDecisionFilter(df)}
                  className={[
                    "px-1.5 py-0.5 rounded text-[11px] font-semibold cursor-pointer text-center transition-colors",
                    decisionFilter === df
                      ? df === "included"
                        ? "bg-blue-700 text-white"
                        : df === "excluded"
                          ? "bg-rose-700 text-white"
                          : "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100",
                  ].join(" ")}
                >
                  {label} ({decisionCounts[df]})
                </button>
              ))}
            </div>

            {/* Unified Multi-Select Flags Bar (audit + codebook flags together) */}
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2 px-0.5 text-[10px]">
                <span className="font-semibold uppercase tracking-wider text-slate-500">
                  Flags{activeFlags.length > 0 ? ` (${activeFlags.length} active)` : ""}
                </span>
                {hasActiveUnifiedFilter && (
                  <button
                    type="button"
                    onClick={resetUnifiedFilters}
                    className="text-blue-700 font-semibold hover:underline cursor-pointer"
                  >
                    Clear filters
                  </button>
                )}
              </div>
              <div
                className="flex flex-wrap gap-1"
                role="group"
                aria-label="Question and codebook flags"
              >
                {visibleFlags.map((flag) => {
                  const active = activeFlags.includes(flag);
                  const count = flagCounts[flag];
                  if (
                    (flag === "blocking" ||
                      flag === "unsaved" ||
                      flag === "overridden" ||
                      flag === "tagged" ||
                      flag === "option_relabeled") &&
                    flagAnyCount[flag] === 0 &&
                    !active
                  ) {
                    return null;
                  }
                  const activeCls =
                    flag === "blocking"
                      ? "bg-red-600 hover:bg-red-700 text-white border-transparent"
                      : flag === "suspect"
                        ? "bg-amber-500 hover:bg-amber-600 text-white border-transparent"
                        : "bg-slate-800 hover:bg-slate-900 text-white border-transparent";
                  const inactiveCls =
                    count === 0
                      ? "bg-white text-slate-400 border-slate-200 hover:bg-slate-100"
                      : "bg-white text-slate-600 border-slate-200 hover:bg-slate-100";
                  return (
                    <Button
                      key={flag}
                      type="button"
                      variant={active ? "default" : "outline"}
                      size="xs"
                      aria-pressed={active}
                      onClick={() => toggleFlag(flag)}
                      className={[
                        "h-5.5 px-1.5 text-[10.5px] font-semibold cursor-pointer shadow-none",
                        active ? activeCls : inactiveCls,
                      ].join(" ")}
                    >
                      {FLAG_LABELS[flag]} ({count})
                    </Button>
                  );
                })}
              </div>
            </div>

            {/* Single-Select Categories Chip Bar (click active chip again to clear) */}
            {hasCategoryCol && categoryOptions.length > 0 && (
              <div className="space-y-1 pt-0.5">
                <div className="flex items-center justify-between gap-2 px-0.5 text-[10px]">
                  <span className="font-semibold uppercase tracking-wider text-slate-500">
                    Categories
                  </span>
                  {categoryFilter.length > 0 && (
                    <button
                      type="button"
                      onClick={clearCategories}
                      className="text-teal-700 font-semibold hover:underline cursor-pointer"
                    >
                      Clear
                    </button>
                  )}
                </div>
                <div
                  className="flex flex-wrap gap-1 max-h-36 overflow-y-auto pr-0.5"
                  role="group"
                  aria-label="Question categories"
                >
                  {categoryOptions.map(([cat]) => {
                    const active = categoryFilter.includes(cat);
                    const count = categoryCounts[cat] ?? 0;
                    const label =
                      cat === UNCATEGORIZED_CATEGORY ? "Uncategorized" : cat;
                    return (
                      <Button
                        key={cat}
                        type="button"
                        variant={active ? "default" : "outline"}
                        size="xs"
                        aria-pressed={active}
                        onClick={() => toggleCategory(cat)}
                        className={[
                          "h-5.5 px-1.5 text-[10.5px] font-semibold cursor-pointer shadow-none",
                          active
                            ? "bg-teal-700 hover:bg-teal-800 text-white border-transparent"
                            : count === 0
                              ? "bg-white text-slate-400 border-slate-200 hover:bg-slate-100"
                              : "bg-white text-slate-600 border-slate-200 hover:bg-teal-50 hover:text-teal-900 hover:border-teal-300",
                        ].join(" ")}
                      >
                        {label} ({count})
                      </Button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Format-skipped visibility toggle */}
            {formatSkippedCount > 0 && (
              <label className="flex items-center justify-between px-2 py-1 rounded-md bg-slate-100/90 border border-slate-200/80 text-[11px] text-slate-600 cursor-pointer select-none">
                <span>
                  Show format-skipped (Demo / DB / TE / Slider / Multi) dimmed
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
              <div className="p-6 text-center text-xs text-slate-400 space-y-2">
                <div>No questions match current filters.</div>
                {hasActiveUnifiedFilter && (
                  <button
                    type="button"
                    onClick={resetUnifiedFilters}
                    className="text-blue-700 hover:underline font-medium cursor-pointer"
                  >
                    Clear filters
                  </button>
                )}
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
                const srcItem = sourceConfigured
                  ? sourceCheck?.items[r.rowKey]
                  : undefined;
                return (
                  <SidebarRow
                    key={r.rowKey}
                    row={r}
                    schemaKind={payload.schemaKind}
                    subItems={subItems}
                    overrideStem={effectiveOverridesByRowKey[r.rowKey]?.stem}
                    sourceItem={srcItem}
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

        {/* Right Content Pane: swaps cleanly between Source comparison+PDF and Questions DetailPane */}
        {activeTab === "source" ? (
          <SourcePane
            dsId={payload.dsId}
            data={sourceCheck}
            loading={sourceCheckLoading}
            error={sourceCheckError}
            row={currentRow}
            savedData={currentSavedRow?.data}
            savedOverride={currentSavedOverride}
            effectiveOverride={currentEffectiveOverride}
            savedTags={currentSavedTags}
            allTagCounts={tagOptions}
            allCategoryCounts={namedCategoryCounts}
            detail={currentDetail}
            schemaKind={payload.schemaKind}
            columns={payload.columns}
            draftResetSeq={draftResetSeq}
            positionIndex={posInFiltered}
            totalFiltered={filteredRows.length}
            hasPrev={prevRowIdx >= 0}
            hasNext={nextRowIdx >= 0}
            marginalPersona={marginalPersona}
            onSelectMarginalPersona={setMarginalPersona}
            marginalsRefreshSeq={personasRefreshSeq}
            onPrev={goPrev}
            onNext={goNext}
            onStageRow={handleStageSingleRow}
            onStageOverride={handleStageOverride}
            onStageTags={handleStageTags}
            onFilterByTag={handleFilterByTag}
            onFilterByCategory={handleFilterByCategory}
            onSaveRow={handleSaveSingleRow}
            onDiscardRow={handleDiscardRow}
            onToggleReviewed={handleToggleReviewed}
            onConfirmAndNext={() => {
              void handleConfirmAndNext();
            }}
            onEditInQuestions={handleEditInQuestions}
          />
        ) : (
          <DetailPane
            key={`${currentRow?.rowKey ?? "empty"}:${draftResetSeq}`}
            dsId={payload.dsId}
            row={currentRow}
            savedData={currentSavedRow?.data}
            savedOverride={currentSavedOverride}
            effectiveOverride={currentEffectiveOverride}
            savedTags={currentSavedTags}
            allTagCounts={tagOptions}
            allCategoryCounts={namedCategoryCounts}
            detail={currentDetail}
            studyCandidates={currentStudyCandidates}
            schemaKind={payload.schemaKind}
            columns={payload.columns}
            positionIndex={posInFiltered}
            totalFiltered={filteredRows.length}
            hasPrev={prevRowIdx >= 0}
            hasNext={nextRowIdx >= 0}
            activeSubItemIdx={selectedSubItemIdx}
            detailTab={detailTab}
            onChangeDetailTab={setDetailTab}
            marginalPersona={marginalPersona}
            onSelectMarginalPersona={setMarginalPersona}
            marginalsRefreshSeq={personasRefreshSeq}
            onSelectSubItemIdx={setSelectedSubItemIdx}
            onPrev={goPrev}
            onNext={goNext}
            onStageRow={handleStageSingleRow}
            onStageOverride={handleStageOverride}
            onStageTags={handleStageTags}
            onFilterByTag={handleFilterByTag}
            onFilterByCategory={handleFilterByCategory}
            onSaveRow={handleSaveSingleRow}
            onDiscardRow={handleDiscardRow}
            onToggleReviewed={handleToggleReviewed}
            onConfirmAndNext={handleConfirmAndNext}
            onViewInSource={handleViewInSource}
          />
        )}
      </div>
      )}
    </div>
  );
}
