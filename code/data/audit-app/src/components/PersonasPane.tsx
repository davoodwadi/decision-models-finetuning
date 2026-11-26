"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Code2,
  Copy,
  Filter,
  Loader2,
  Search,
  Shuffle,
  User,
  Users,
  XIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type {
  PersonaAttributeSummary,
  PersonaCompletenessFilter,
  PersonaRespondentRow,
  PersonasPagePayload,
  RespondentQuestionsPayload,
} from "@/lib/types";

const PAGE_SIZE = 100;

function hashSeed(input: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function deterministicShuffle<T>(items: T[], seedStr: string): T[] {
  const arr = [...items];
  let a = hashSeed(seedStr) || 1;
  const nextRand = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(nextRand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

interface PersonasPaneProps {
  dsId: string;
  refreshSeq: number;
}

export function PersonasPane({ dsId, refreshSeq }: PersonasPaneProps) {
  const [data, setData] = useState<PersonasPagePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [completeness, setCompleteness] =
    useState<PersonaCompletenessFilter>("all");
  const [attrCol, setAttrCol] = useState("");
  const [attrVal, setAttrVal] = useState("");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(1);

  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [schemaOpen, setSchemaOpen] = useState(true);

  // Live KEV prompt controls
  const [includeCountry, setIncludeCountry] = useState(true);
  const [orderMode, setOrderMode] = useState<"canonical" | "shuffled">(
    "canonical"
  );
  const [shuffleSeed, setShuffleSeed] = useState(0);
  const [showRawViewJson, setShowRawViewJson] = useState(false);
  const [copiedPrompt, setCopiedPrompt] = useState(false);

  // Selected respondent's answered questions
  const [respQuestions, setRespQuestions] =
    useState<RespondentQuestionsPayload | null>(null);
  const [loadingQuestions, setLoadingQuestions] = useState(false);
  const [questionsError, setQuestionsError] = useState<string | null>(null);
  const [selectedQuestionIdx, setSelectedQuestionIdx] = useState(0);
  const questionsCacheRef = useRef<Map<string, RespondentQuestionsPayload>>(
    new Map()
  );

  // Reset filters & caches when switching dataset or after parquet rebuild
  useEffect(() => {
    setCompleteness("all");
    setAttrCol("");
    setAttrVal("");
    setSearch("");
    setDebouncedSearch("");
    setPage(1);
    setSelectedCaseId(null);
    setRespQuestions(null);
    setSelectedQuestionIdx(0);
    questionsCacheRef.current.clear();
  }, [dsId, refreshSeq]);

  // Debounce search text
  useEffect(() => {
    const id = window.setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 150);
    return () => window.clearTimeout(id);
  }, [search]);

  // Fetch paginated personas summary
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    const sp = new URLSearchParams({
      ds: dsId,
      completeness,
      page: String(page),
      pageSize: String(PAGE_SIZE),
    });
    if (debouncedSearch.trim()) sp.set("search", debouncedSearch.trim());
    if (attrCol && attrVal !== "") {
      sp.set("attrCol", attrCol);
      sp.set("attrVal", attrVal);
    }

    fetch(`/api/personas?${sp.toString()}`, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (res) => {
        const json = await res.json();
        if (!res.ok) {
          throw new Error(json.error || "Failed to load personas");
        }
        return json as PersonasPagePayload;
      })
      .then((payload) => {
        setData(payload);
        setSelectedCaseId((prev) => {
          if (prev && payload.respondents.some((r) => r.caseId === prev)) {
            return prev;
          }
          return payload.respondents[0]?.caseId ?? null;
        });
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        setError(err instanceof Error ? err.message : String(err));
        setData(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [dsId, refreshSeq, completeness, attrCol, attrVal, debouncedSearch, page]);

  // Fetch answered questions for the currently selected respondent
  useEffect(() => {
    if (!selectedCaseId) {
      setRespQuestions(null);
      setQuestionsError(null);
      return;
    }
    const cacheKey = `${dsId}::${refreshSeq}::${selectedCaseId}`;
    const cached = questionsCacheRef.current.get(cacheKey);
    if (cached) {
      setRespQuestions(cached);
      setSelectedQuestionIdx(0);
      setQuestionsError(null);
      return;
    }

    const controller = new AbortController();
    setLoadingQuestions(true);
    setQuestionsError(null);
    const sp = new URLSearchParams({
      ds: dsId,
      caseId: selectedCaseId,
      limit: "50",
    });

    fetch(`/api/personas/questions?${sp.toString()}`, {
      cache: "no-store",
      signal: controller.signal,
    })
      .then(async (res) => {
        const json = await res.json();
        if (!res.ok) {
          throw new Error(json.error || "Failed to load respondent questions");
        }
        return json as RespondentQuestionsPayload;
      })
      .then((payload) => {
        questionsCacheRef.current.set(cacheKey, payload);
        setRespQuestions(payload);
        setSelectedQuestionIdx(0);
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        setQuestionsError(err instanceof Error ? err.message : String(err));
        setRespQuestions(null);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingQuestions(false);
      });

    return () => controller.abort();
  }, [dsId, refreshSeq, selectedCaseId]);

  const respondents = data?.respondents ?? [];
  const selectedIdx = useMemo(
    () =>
      selectedCaseId
        ? respondents.findIndex((r) => r.caseId === selectedCaseId)
        : -1,
    [respondents, selectedCaseId]
  );
  const selectedRespondent: PersonaRespondentRow | undefined =
    selectedIdx >= 0 ? respondents[selectedIdx] : undefined;

  const goPrevResp = useCallback(() => {
    if (selectedIdx > 0) {
      setSelectedCaseId(respondents[selectedIdx - 1].caseId);
    }
  }, [selectedIdx, respondents]);

  const goNextResp = useCallback(() => {
    if (selectedIdx >= 0 && selectedIdx < respondents.length - 1) {
      setSelectedCaseId(respondents[selectedIdx + 1].caseId);
    }
  }, [selectedIdx, respondents]);

  // j / k or ArrowUp / ArrowDown navigation through respondents
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        goNextResp();
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        goPrevResp();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [goNextResp, goPrevResp]);

  const activeAttrMeta: PersonaAttributeSummary | undefined = useMemo(
    () => data?.attributes.find((a) => a.col === attrCol),
    [data, attrCol]
  );

  const handleClickLevelChip = useCallback(
    (col: string, val: string) => {
      if (attrCol === col && attrVal === val) {
        setAttrCol("");
        setAttrVal("");
      } else {
        setAttrCol(col);
        setAttrVal(val);
      }
      setPage(1);
    },
    [attrCol, attrVal]
  );

  // Build exact KEV state lines matching `data.base.make_view()`
  const kevStateInfo = useMemo(() => {
    if (!data || !selectedRespondent) {
      return {
        emitted: [] as { col: string; label: string; value: string; line: string }[],
        omittedMissing: [] as { col: string; label: string }[],
        omittedCountry: false,
      };
    }
    const emitted: { col: string; label: string; value: string; line: string }[] = [];
    const omittedMissing: { col: string; label: string }[] = [];
    let omittedCountry = false;

    for (const attr of data.attributes) {
      const rawVal = selectedRespondent.values[attr.col] ?? null;
      if (attr.isCountry && !includeCountry) {
        omittedCountry = true;
        continue;
      }
      if (rawVal === null) {
        omittedMissing.push({ col: attr.col, label: attr.promptLabel });
        continue;
      }
      emitted.push({
        col: attr.col,
        label: attr.promptLabel,
        value: rawVal,
        line: `${attr.promptLabel}: ${rawVal}`,
      });
    }

    const ordered =
      orderMode === "shuffled" && emitted.length > 1
        ? deterministicShuffle(
            emitted,
            `${dsId}:${selectedRespondent.caseId}:${shuffleSeed}`
          )
        : emitted;

    return { emitted: ordered, omittedMissing, omittedCountry };
  }, [data, selectedRespondent, includeCountry, orderMode, dsId, shuffleSeed]);

  const safeQuestionIdx = respQuestions?.questions.length
    ? Math.min(selectedQuestionIdx, respQuestions.questions.length - 1)
    : 0;
  const activeQuestion = respQuestions?.questions[safeQuestionIdx];

  const makeViewJson = useMemo(() => {
    const stateText = kevStateInfo.emitted.map((e) => e.line).join("\n");
    if (!activeQuestion) {
      return { state: stateText, questions: {} };
    }
    if (activeQuestion.type === "score") {
      const target: Record<string, number> = {};
      activeQuestion.optionTexts.forEach((_, i) => {
        target[String(i)] = activeQuestion.targetProbs
          ? activeQuestion.targetProbs[i]
          : i === activeQuestion.answerIdx
            ? 1.0
            : 0.0;
      });
      return {
        state: stateText,
        questions: {
          [activeQuestion.qid]: {
            type: "score",
            instructions: activeQuestion.question,
            criteria: activeQuestion.optionTexts,
            target,
          },
        },
      };
    }
    const criteria: Record<string, string | null> = {};
    const target: Record<string, number> = {};
    activeQuestion.optionTexts.forEach((opt, i) => {
      criteria[opt] = activeQuestion.optionDescriptions
        ? (activeQuestion.optionDescriptions[i] ?? null)
        : null;
      target[opt] = activeQuestion.targetProbs
        ? activeQuestion.targetProbs[i]
        : i === activeQuestion.answerIdx
          ? 1.0
          : 0.0;
    });
    return {
      state: stateText,
      questions: {
        [activeQuestion.qid]: {
          type: "choice",
          instructions: activeQuestion.question,
          criteria,
          target,
        },
      },
    };
  }, [kevStateInfo.emitted, activeQuestion]);

  const formattedPlainKevPrompt = useMemo(() => {
    const stateBlock = kevStateInfo.emitted.map((e) => e.line).join("\n");
    const lines = ["<|fim_prefix|>", stateBlock];
    if (activeQuestion) {
      lines.push("<|fim_middle|>", activeQuestion.question);
      activeQuestion.optionTexts.forEach((opt, i) => {
        const desc =
          activeQuestion.type === "choice" && activeQuestion.optionDescriptions
            ? activeQuestion.optionDescriptions[i]
            : null;
        const optRendered = desc ? `${opt}: ${desc}` : opt;
        lines.push(`<|box_start|>${optRendered}<|box_end|>`);
      });
      lines.push("<|fim_suffix|>");
    }
    return lines.join("\n");
  }, [kevStateInfo.emitted, activeQuestion]);

  const handleCopyPrompt = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(formattedPlainKevPrompt);
      setCopiedPrompt(true);
      window.setTimeout(() => setCopiedPrompt(false), 1600);
    } catch {
      // Clipboard unavailable
    }
  }, [formattedPlainKevPrompt]);

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 bg-slate-50">
        <div className="max-w-lg rounded-xl border border-rose-200 bg-white p-5 space-y-2 shadow-2xs">
          <div className="text-sm font-semibold text-rose-800">
            Could not load personas for{" "}
            <code className="font-mono">{dsId}</code>
          </div>
          <p className="text-xs text-slate-600 font-mono whitespace-pre-wrap">
            {error}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex overflow-hidden">
      {/* Left Sidebar: Respondent Browser */}
      <aside className="w-[390px] shrink-0 bg-white border-r border-slate-200 flex flex-col h-full overflow-hidden">
        <div className="p-2.5 border-b border-slate-200 space-y-2 bg-slate-50/60">
          {/* Completeness Tabs */}
          <div className="flex flex-wrap gap-1">
            {(
              [
                ["all", "All", data?.nRespondents ?? 0],
                ["complete", "Complete", data?.completeCount ?? 0],
                ["incomplete", "Has NA", data?.incompleteCount ?? 0],
              ] as [PersonaCompletenessFilter, string, number][]
            ).map(([mode, label, count]) => {
              const active = completeness === mode;
              return (
                <Button
                  key={mode}
                  type="button"
                  variant={active ? "default" : "outline"}
                  size="xs"
                  onClick={() => {
                    setCompleteness(mode);
                    setPage(1);
                  }}
                  className={[
                    "text-[11px] font-semibold cursor-pointer shadow-none",
                    active
                      ? mode === "incomplete"
                        ? "bg-amber-500 hover:bg-amber-600 text-white border-transparent"
                        : "bg-slate-900 hover:bg-slate-800 text-white border-transparent"
                      : "bg-white text-slate-600 border-slate-200 hover:bg-slate-100",
                  ].join(" ")}
                >
                  {label} ({count.toLocaleString()})
                </Button>
              );
            })}
          </div>

          {/* Attribute & Level Filter */}
          <div className="grid grid-cols-2 gap-1.5">
            <Select
              value={attrCol || "__all__"}
              onValueChange={(val) => {
                const nextCol = !val || val === "__all__" ? "" : val;
                setAttrCol(nextCol);
                setAttrVal("");
                setPage(1);
              }}
            >
              <SelectTrigger
                size="sm"
                className="w-full bg-white text-xs text-slate-700 shadow-none"
              >
                <SelectValue>
                  {(val: string | null) => {
                    if (!val || val === "__all__") {
                      return `Filter attr (${data?.nAttributes ?? 0})`;
                    }
                    return val;
                  }}
                </SelectValue>
              </SelectTrigger>
              <SelectContent align="start">
                <SelectItem value="__all__" className="text-xs">
                  All attributes
                </SelectItem>
                {(data?.attributes ?? []).map((a) => (
                  <SelectItem key={a.col} value={a.col} className="text-xs">
                    {a.col} ({a.nLevels})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <Select
              value={attrVal || "__all__"}
              disabled={!attrCol}
              onValueChange={(val) => {
                setAttrVal(!val || val === "__all__" ? "" : val);
                setPage(1);
              }}
            >
              <SelectTrigger
                size="sm"
                className="w-full bg-white text-xs text-slate-700 shadow-none"
              >
                <SelectValue>
                  {(val: string | null) => {
                    if (!attrCol) return "Pick attribute first";
                    if (!val || val === "__all__") return "Any value";
                    if (val === "__MISSING__") return "Missing (NA)";
                    return val;
                  }}
                </SelectValue>
              </SelectTrigger>
              <SelectContent align="start">
                <SelectItem value="__all__" className="text-xs">
                  Any value
                </SelectItem>
                {activeAttrMeta && activeAttrMeta.nMissing > 0 && (
                  <SelectItem value="__MISSING__" className="text-xs text-amber-700">
                    Missing (NA) ({activeAttrMeta.nMissing})
                  </SelectItem>
                )}
                {(activeAttrMeta?.levels ?? []).map((lv) => (
                  <SelectItem
                    key={lv.value}
                    value={lv.value}
                    className="text-xs"
                  >
                    {lv.value} ({lv.count})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Search Box */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            <Input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search CASEID, source ID, or trait value..."
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

          {/* Pagination Row */}
          {data && (
            <div className="flex items-center justify-between text-[11px] text-slate-500 pt-0.5">
              <span>
                {data.totalFiltered === 0
                  ? "0 respondents"
                  : `${((data.page - 1) * data.pageSize + 1).toLocaleString()}–${Math.min(
                      data.page * data.pageSize,
                      data.totalFiltered
                    ).toLocaleString()} of ${data.totalFiltered.toLocaleString()}`}
              </span>
              <div className="inline-flex items-center gap-1">
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  disabled={data.page <= 1 || loading}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="h-5 px-1.5 bg-white cursor-pointer"
                >
                  <ChevronLeft className="w-3 h-3" />
                </Button>
                <span className="font-mono text-[11px] px-1">
                  {data.page}/{data.totalPages}
                </span>
                <Button
                  type="button"
                  variant="outline"
                  size="xs"
                  disabled={data.page >= data.totalPages || loading}
                  onClick={() =>
                    setPage((p) => Math.min(data.totalPages, p + 1))
                  }
                  className="h-5 px-1.5 bg-white cursor-pointer"
                >
                  <ChevronRight className="w-3 h-3" />
                </Button>
              </div>
            </div>
          )}
        </div>

        {/* Scrollable Respondent List */}
        <div className="flex-1 overflow-y-auto divide-y divide-slate-100">
          {loading && !data ? (
            <div className="p-8 flex items-center justify-center gap-2 text-xs text-slate-500">
              <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
              Loading personas...
            </div>
          ) : respondents.length === 0 ? (
            <div className="p-6 text-center text-xs text-slate-400">
              No respondents match current filters.
            </div>
          ) : (
            respondents.map((r) => {
              const isSelected = r.caseId === selectedCaseId;
              const nTotalAttrs = data?.nAttributes ?? 0;
              const isComplete = r.nPresent >= nTotalAttrs;
              const previewTraits = (data?.attributes ?? [])
                .filter((a) => !a.isCountry && r.values[a.col] !== null)
                .slice(0, 5)
                .map((a) => r.values[a.col])
                .join(" · ");
              const extraEntries = Object.entries(r.extraIds).filter(
                ([, v]) => v !== null
              );

              return (
                <button
                  key={r.caseId}
                  type="button"
                  onClick={() => setSelectedCaseId(r.caseId)}
                  className={[
                    "w-full text-left px-3 py-2 transition-colors cursor-pointer",
                    isSelected
                      ? "bg-blue-50/90 border-l-2 border-l-blue-600"
                      : "hover:bg-slate-50 border-l-2 border-l-transparent",
                  ].join(" ")}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="font-mono text-xs font-semibold text-slate-900 truncate">
                        {r.caseId}
                      </span>
                      {extraEntries.slice(0, 1).map(([k, v]) => (
                        <span
                          key={k}
                          className="font-mono text-[10px] text-slate-400 truncate max-w-[115px]"
                          title={`${k}: ${v}`}
                        >
                          {v}
                        </span>
                      ))}
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <span
                        className={[
                          "px-1.5 py-0.5 rounded font-mono text-[10px] font-semibold",
                          isComplete
                            ? "bg-emerald-50 text-emerald-700"
                            : "bg-amber-50 text-amber-800",
                        ].join(" ")}
                        title={`${r.nPresent} of ${nTotalAttrs} persona attributes present`}
                      >
                        {r.nPresent}/{nTotalAttrs}
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-mono text-[10px]">
                        {r.nQuestions} Q
                      </span>
                    </div>
                  </div>
                  <div className="text-[11px] text-slate-500 truncate mt-0.5">
                    {previewTraits || "(no non-country traits)"}
                  </div>
                </button>
              );
            })
          )}
        </div>
      </aside>

      {/* Right Main Pane: Schema Overview + Respondent Detail + Live KEV Prompt */}
      <main className="flex-1 overflow-y-auto bg-slate-50/70 p-5 space-y-4">
        {loading && !data ? (
          <div className="flex items-center justify-center h-64 gap-2 text-sm text-slate-500">
            <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
            Reading parquet personas...
          </div>
        ) : data ? (
          <>
            {/* Dataset Persona Summary Strip */}
            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs space-y-3">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-2 flex-wrap">
                  <Users className="w-4 h-4 text-blue-600" />
                  <h2 className="text-sm font-bold text-slate-900">
                    Dataset Personas
                  </h2>
                  <Badge
                    variant="outline"
                    className="font-mono text-[11px] bg-slate-50 text-slate-700"
                  >
                    {data.parquetFile}
                  </Badge>
                  <Badge
                    variant="outline"
                    className="font-mono text-[11px] bg-blue-50 text-blue-800 border-blue-200"
                    title="Exact Python loader function determining canonical persona attribute order"
                  >
                    {data.loaderKind}
                  </Badge>
                  {data.isspRuntimeMapped && (
                    <Badge
                      variant="outline"
                      className="text-[11px] bg-violet-50 text-violet-800 border-violet-200"
                      title="ISSP parquets store raw codes; apply_persona_mapping(df, dataset='issp') is applied at load time"
                    >
                      ISSP runtime-mapped
                    </Badge>
                  )}
                  {data.isMarginal && (
                    <Badge
                      variant="outline"
                      className="text-[11px] bg-amber-50 text-amber-800 border-amber-200"
                      title="Aggregate subgroup marginals (no respondent-level microdata)"
                    >
                      Aggregate marginals
                    </Badge>
                  )}
                </div>
                <span className="text-[11px] text-slate-500">
                  {data.nRows.toLocaleString()} {data.isMarginal ? "marginal rows" : "long rows"} ·{" "}
                  {data.nQuestions.toLocaleString()} unique QIDs
                </span>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    {data.isMarginal ? "Subgroups" : "Respondents"}
                  </div>
                  <div className="text-base font-bold text-slate-900 font-mono">
                    {data.nRespondents.toLocaleString()}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {data.isMarginal
                      ? "1 active trait + Country per subgroup"
                      : `${data.completeCount.toLocaleString()} complete · ${data.incompleteCount.toLocaleString()} with NA`}
                  </div>
                </div>

                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    Persona Attributes
                  </div>
                  <div className="text-base font-bold text-slate-900 font-mono">
                    {data.nAttributes}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {data.hasCountry
                      ? `c_alphan + ${data.nAttributes - 1} traits`
                      : `${data.nAttributes} traits (no country)`}
                  </div>
                </div>

                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    Unique Profiles
                  </div>
                  <div className="text-base font-bold text-slate-900 font-mono">
                    {(includeCountry
                      ? data.uniquePersonasWithCountry
                      : data.uniquePersonasWithoutCountry
                    ).toLocaleString()}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    {includeCountry
                      ? "with Country (c_alphan)"
                      : "without Country"}
                  </div>
                </div>

                <div className="rounded-lg bg-slate-50 border border-slate-200/80 px-3 py-2">
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                    Traits / Respondent
                  </div>
                  <div className="text-base font-bold text-slate-900 font-mono">
                    {data.avgAttrsPresent.toFixed(2)} / {data.nAttributes}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    range {data.minAttrsPresent}–{data.maxAttrsPresent} present
                  </div>
                </div>
              </div>
            </div>

            {/* Selected Respondent Section */}
            {selectedRespondent ? (
              <>
                {/* Card 2: Selected Respondent Structured Persona Mapping */}
                <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs space-y-3">
                  <div className="flex items-center justify-between gap-3 flex-wrap">
                    <div className="flex items-center gap-2 flex-wrap">
                      <User className="w-4 h-4 text-slate-700" />
                      <h3 className="text-sm font-bold text-slate-900 font-mono">
                        CASEID: {selectedRespondent.caseId}
                      </h3>
                      {Object.entries(selectedRespondent.extraIds).map(
                        ([k, v]) =>
                          v !== null ? (
                            <Badge
                              key={k}
                              variant="outline"
                              className="font-mono text-[11px] bg-slate-50 text-slate-600"
                            >
                              {k}: {v}
                            </Badge>
                          ) : null
                      )}
                      <Badge
                        variant="outline"
                        className={[
                          "font-mono text-[11px]",
                          selectedRespondent.nPresent >= data.nAttributes
                            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                            : "bg-amber-50 text-amber-800 border-amber-200",
                        ].join(" ")}
                      >
                        {selectedRespondent.nPresent}/{data.nAttributes} traits
                      </Badge>
                      <Badge
                        variant="outline"
                        className="font-mono text-[11px] bg-slate-50 text-slate-700"
                      >
                        {selectedRespondent.nQuestions} answered Qs
                      </Badge>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] text-slate-400 font-mono mr-1">
                        Row {selectedIdx + 1}/{respondents.length} on page
                      </span>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        disabled={selectedIdx <= 0}
                        onClick={goPrevResp}
                        className="bg-white cursor-pointer"
                        title="Previous respondent (k)"
                      >
                        <ChevronLeft className="w-3.5 h-3.5" />
                        Prev
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="xs"
                        disabled={
                          selectedIdx < 0 ||
                          selectedIdx >= respondents.length - 1
                        }
                        onClick={goNextResp}
                        className="bg-white cursor-pointer"
                        title="Next respondent (j)"
                      >
                        Next
                        <ChevronRight className="w-3.5 h-3.5" />
                      </Button>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs border-collapse">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-400 text-left">
                          <th className="py-1.5 pr-2 w-8 font-medium">#</th>
                          <th className="py-1.5 pr-3 w-32 font-medium">Key</th>
                          <th className="py-1.5 pr-3 w-64 font-medium">
                            KEV Prompt Label
                          </th>
                          <th className="py-1.5 pr-3 w-52 font-medium">
                            Respondent Value
                          </th>
                          <th className="py-1.5 font-medium">
                            Emitted KEV State Line
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {data.attributes.map((attr, idx) => {
                          const val = selectedRespondent.values[attr.col] ?? null;
                          const rawVal = selectedRespondent.rawValues?.[attr.col];
                          const hiddenByCountry =
                            attr.isCountry && !includeCountry;
                          const isMissing = val === null;
                          return (
                            <tr
                              key={attr.col}
                              className={
                                isMissing || hiddenByCountry
                                  ? "bg-slate-50/70 text-slate-400"
                                  : ""
                              }
                            >
                              <td className="py-1.5 pr-2 font-mono text-slate-400">
                                {idx + 1}
                              </td>
                              <td className="py-1.5 pr-3 font-mono font-semibold text-slate-800">
                                {attr.col}
                              </td>
                              <td className="py-1.5 pr-3 text-slate-600">
                                {attr.promptLabel}
                              </td>
                              <td className="py-1.5 pr-3">
                                {isMissing ? (
                                  <span className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 font-mono text-[10px]">
                                    NaN (missing)
                                  </span>
                                ) : (
                                  <div className="flex items-center gap-1.5 flex-wrap">
                                    <span className="font-medium text-slate-900">
                                      {val}
                                    </span>
                                    {rawVal !== undefined && (
                                      <span
                                        className="px-1.5 py-0 rounded bg-violet-50 text-violet-700 border border-violet-200 font-mono text-[10px]"
                                        title="Raw value stored in parquet before apply_persona_mapping"
                                      >
                                        raw: {rawVal ?? "NaN"}
                                      </span>
                                    )}
                                  </div>
                                )}
                              </td>
                              <td className="py-1.5 font-mono text-[11px]">
                                {hiddenByCountry ? (
                                  <span className="italic text-slate-400">
                                    omitted (include_country=False)
                                  </span>
                                ) : isMissing ? (
                                  <span className="italic text-amber-700">
                                    omitted from state (NaN)
                                  </span>
                                ) : (
                                  <span className="text-emerald-800 bg-emerald-50/70 px-1.5 py-0.5 rounded">
                                    {attr.promptLabel}: {val}
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Card 3: Live KEV Prompt Preview (Full <|fim_prefix|> State + Answered Question) */}
                <div className="bg-slate-900 text-slate-100 rounded-xl p-4 shadow-xs font-mono text-xs leading-relaxed overflow-x-auto space-y-3">
                  {/* Dark Header Controls */}
                  <div className="flex items-center justify-between gap-2 flex-wrap font-sans">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
                        Live Kev Prompt Preview (Respondent State + Question)
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/20 text-amber-200">
                        {kevStateInfo.emitted.length} state lines
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 flex-wrap">
                      {data.hasCountry && (
                        <button
                          type="button"
                          onClick={() => setIncludeCountry((v) => !v)}
                          title="Toggle include_country in build_long_records / build_persona_dataset (False in US-only beta training)"
                          className={[
                            "px-2 py-1 rounded text-[11px] font-mono border transition-colors cursor-pointer",
                            includeCountry
                              ? "bg-sky-500/20 text-sky-200 border-sky-400/40"
                              : "bg-slate-800 text-slate-400 border-slate-700 hover:text-slate-200",
                          ].join(" ")}
                        >
                          include_country={includeCountry ? "True" : "False"}
                        </button>
                      )}

                      <div className="inline-flex rounded border border-slate-700 bg-slate-800 p-0.5 text-[11px]">
                        <button
                          type="button"
                          onClick={() => setOrderMode("canonical")}
                          className={[
                            "px-2 py-0.5 rounded transition-colors cursor-pointer",
                            orderMode === "canonical"
                              ? "bg-slate-600 text-white font-semibold"
                              : "text-slate-400 hover:text-slate-200",
                          ].join(" ")}
                        >
                          Canonical (eval)
                        </button>
                        <button
                          type="button"
                          onClick={() => setOrderMode("shuffled")}
                          className={[
                            "px-2 py-0.5 rounded transition-colors cursor-pointer",
                            orderMode === "shuffled"
                              ? "bg-indigo-600 text-white font-semibold"
                              : "text-slate-400 hover:text-slate-200",
                          ].join(" ")}
                        >
                          Shuffled (train)
                        </button>
                      </div>

                      {orderMode === "shuffled" && (
                        <button
                          type="button"
                          onClick={() => setShuffleSeed((s) => s + 1)}
                          className="inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] bg-slate-800 border border-slate-700 text-indigo-200 hover:bg-slate-700 cursor-pointer"
                          title="Simulate another random attribute permutation from make_view(rec, rng)"
                        >
                          <Shuffle className="w-3 h-3" />
                          Reshuffle
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => setShowRawViewJson((v) => !v)}
                        className={[
                          "inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] border cursor-pointer",
                          showRawViewJson
                            ? "bg-emerald-500/20 text-emerald-200 border-emerald-400/40"
                            : "bg-slate-800 text-slate-300 border-slate-700 hover:text-white",
                        ].join(" ")}
                      >
                        <Code2 className="w-3 h-3" />
                        make_view() JSON
                      </button>

                      <button
                        type="button"
                        onClick={() => void handleCopyPrompt()}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded text-[11px] bg-slate-800 border border-slate-700 text-slate-200 hover:bg-slate-700 cursor-pointer"
                      >
                        {copiedPrompt ? (
                          <Check className="w-3 h-3 text-emerald-400" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                        {copiedPrompt ? "Copied" : "Copy"}
                      </button>
                    </div>
                  </div>

                  {/* Sample Question Picker for this Respondent */}
                  <div className="flex items-center gap-2 flex-wrap font-sans bg-slate-800/80 border border-slate-700/80 rounded-lg px-2.5 py-1.5">
                    <span className="text-[11px] text-slate-400 shrink-0">
                      Answered question:
                    </span>
                    {loadingQuestions ? (
                      <span className="inline-flex items-center gap-1.5 text-[11px] text-slate-300">
                        <Loader2 className="w-3.5 h-3.5 animate-spin text-blue-400" />
                        Loading respondent questions...
                      </span>
                    ) : questionsError ? (
                      <span className="text-[11px] text-rose-300">
                        {questionsError}
                      </span>
                    ) : respQuestions && respQuestions.questions.length > 0 ? (
                      <>
                        <div className="inline-flex items-center gap-1 shrink-0">
                          <button
                            type="button"
                            disabled={safeQuestionIdx <= 0}
                            onClick={() =>
                              setSelectedQuestionIdx((i) => Math.max(0, i - 1))
                            }
                            className="px-1 text-slate-300 hover:text-white disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                          >
                            ◀
                          </button>
                          <span className="font-mono text-[11px] text-indigo-300">
                            {safeQuestionIdx + 1}/
                            {respQuestions.questions.length}
                            {respQuestions.totalQuestions >
                            respQuestions.questions.length
                              ? ` (of ${respQuestions.totalQuestions})`
                              : ""}
                          </span>
                          <button
                            type="button"
                            disabled={
                              safeQuestionIdx >=
                              respQuestions.questions.length - 1
                            }
                            onClick={() =>
                              setSelectedQuestionIdx((i) =>
                                Math.min(
                                  respQuestions.questions.length - 1,
                                  i + 1
                                )
                              )
                            }
                            className="px-1 text-slate-300 hover:text-white disabled:opacity-35 cursor-pointer disabled:cursor-not-allowed"
                          >
                            ▶
                          </button>
                        </div>
                        <select
                          value={safeQuestionIdx}
                          onChange={(e) =>
                            setSelectedQuestionIdx(Number(e.target.value))
                          }
                          className="flex-1 min-w-[200px] h-6 rounded bg-slate-900 border border-slate-700 px-2 text-[11px] font-mono text-slate-200"
                        >
                          {respQuestions.questions.map((q, i) => (
                            <option key={`${q.qid}-${i}`} value={i}>
                              [{q.type}] {q.qid} —{" "}
                              {q.question.replace(/\s+/g, " ").slice(0, 80)}
                            </option>
                          ))}
                        </select>
                        {activeQuestion && (
                          <span
                            className={[
                              "px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase shrink-0",
                              activeQuestion.type === "score"
                                ? "bg-violet-500/25 text-violet-200"
                                : "bg-blue-500/25 text-blue-200",
                            ].join(" ")}
                          >
                            {activeQuestion.type}
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="text-[11px] text-slate-400">
                        No questions loaded
                      </span>
                    )}
                  </div>

                  {/* Exact KEV Token Stream */}
                  <div className="space-y-1">
                    <div>
                      <span className="text-amber-400">&lt;|fim_prefix|&gt;</span>
                      <span className="ml-2 text-[10px] text-slate-500 font-sans">
                        (persona state)
                      </span>
                    </div>
                    {kevStateInfo.emitted.length > 0 ? (
                      <div className="pl-2 border-l-2 border-amber-500/40 space-y-0.5 text-amber-50">
                        {kevStateInfo.emitted.map((e) => (
                          <div key={e.col} className="whitespace-pre-wrap">
                            <span className="text-amber-300/90">{e.label}:</span>{" "}
                            <span>{e.value}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="pl-2 text-slate-500 italic">
                        (empty state)
                      </div>
                    )}

                    {(kevStateInfo.omittedMissing.length > 0 ||
                      kevStateInfo.omittedCountry) && (
                      <div className="text-[11px] font-sans text-slate-400 pt-0.5">
                        Omitted from state:{" "}
                        {[
                          ...(kevStateInfo.omittedCountry
                            ? ["Country (include_country=False)"]
                            : []),
                          ...kevStateInfo.omittedMissing.map(
                            (m) => `${m.col} (NaN)`
                          ),
                        ].join(", ")}
                      </div>
                    )}

                    {activeQuestion && (
                      <div className="pt-2 space-y-1">
                        <div>
                          <span className="text-emerald-400">
                            &lt;|fim_middle|&gt;
                          </span>
                          <span className="ml-2 text-[10px] text-slate-500 font-sans">
                            (question branch: {activeQuestion.qid})
                          </span>
                        </div>
                        <div className="whitespace-pre-wrap text-slate-100 my-1">
                          {activeQuestion.question}
                        </div>
                        <div className="space-y-0.5">
                          {activeQuestion.optionTexts.map((opt, i) => {
                            const desc =
                              activeQuestion.type === "choice" &&
                              activeQuestion.optionDescriptions
                                ? activeQuestion.optionDescriptions[i]
                                : null;
                            const isChosen = i === activeQuestion.answerIdx;
                            return (
                              <div
                                key={`${i}-${opt}`}
                                className={
                                  isChosen
                                    ? "bg-emerald-500/10 rounded px-1 -mx-1"
                                    : ""
                                }
                              >
                                <span className="text-sky-400">
                                  &lt;|box_start|&gt;
                                </span>
                                <span className="text-slate-100">
                                  {desc ? `${opt}: ${desc}` : opt}
                                </span>
                                <span className="text-sky-400">
                                  &lt;|box_end|&gt;
                                </span>
                                {activeQuestion.targetProbs ? (
                                  <span
                                    className={[
                                      "ml-2 px-1.5 py-0.5 rounded font-mono text-[10px] font-semibold",
                                      isChosen
                                        ? "bg-emerald-500/25 text-emerald-200"
                                        : "bg-slate-700/80 text-slate-300",
                                    ].join(" ")}
                                  >
                                    {(activeQuestion.targetProbs[i] * 100).toFixed(1)}%
                                    {isChosen ? " (mode)" : ""}
                                  </span>
                                ) : (
                                  isChosen && (
                                    <span className="ml-2 px-1.5 py-0.5 rounded bg-emerald-500/25 text-emerald-200 font-sans text-[10px] font-semibold">
                                      ✓ human answer
                                    </span>
                                  )
                                )}
                              </div>
                            );
                          })}
                        </div>
                        <div className="text-emerald-400 pt-1">
                          &lt;|fim_suffix|&gt;
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Optional Raw make_view() JSON */}
                  {showRawViewJson && (
                    <div className="border-t border-slate-800 pt-3">
                      <div className="text-[10px] uppercase tracking-wider text-slate-400 font-sans font-semibold mb-1.5">
                        Exact data.base.make_view(rec) Output
                      </div>
                      <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-[11px] text-slate-200 overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(makeViewJson, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="bg-white rounded-xl border border-slate-200 p-8 text-center text-xs text-slate-400">
                Select a respondent on the left to inspect their persona and
                live KEV prompt.
              </div>
            )}

            {/* Collapsible Card: Dataset Persona Schema & Level Distributions */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
              <div className="px-4 py-2.5 border-b border-slate-200 flex items-center justify-between gap-2 bg-slate-50/70">
                <button
                  type="button"
                  onClick={() => setSchemaOpen((v) => !v)}
                  className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-600 hover:text-slate-900 cursor-pointer"
                >
                  {schemaOpen ? (
                    <ChevronUp className="w-3.5 h-3.5" />
                  ) : (
                    <ChevronDown className="w-3.5 h-3.5" />
                  )}
                  Persona Schema &amp; Level Distributions ({data.nAttributes}{" "}
                  attributes in canonical KEV order)
                </button>
                {attrCol && attrVal !== "" && (
                  <div className="flex items-center gap-1.5">
                    <Badge
                      variant="outline"
                      className="text-[11px] bg-blue-50 text-blue-800 border-blue-200 gap-1"
                    >
                      <Filter className="w-3 h-3" />
                      {attrCol} ={" "}
                      {attrVal === "__MISSING__" ? "Missing (NA)" : attrVal}
                    </Badge>
                    <Button
                      type="button"
                      variant="ghost"
                      size="xs"
                      onClick={() => {
                        setAttrCol("");
                        setAttrVal("");
                        setPage(1);
                      }}
                      className="h-5 px-1.5 text-[11px] text-slate-500 cursor-pointer"
                    >
                      Clear
                    </Button>
                  </div>
                )}
              </div>

              {schemaOpen && (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 bg-slate-50/40 text-slate-400 text-left">
                        <th className="py-2 pl-4 pr-2 w-8 font-medium">#</th>
                        <th className="py-2 pr-3 w-36 font-medium">
                          Column Key
                        </th>
                        <th className="py-2 pr-3 w-64 font-medium">
                          KEV Prompt Label
                        </th>
                        <th className="py-2 pr-3 w-32 font-medium">Coverage</th>
                        <th className="py-2 pr-4 font-medium">
                          Levels (click any value to filter respondents)
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {data.attributes.map((attr, idx) => (
                        <tr
                          key={attr.col}
                          className={[
                            "align-top",
                            attr.isCountry && !includeCountry
                              ? "opacity-55 bg-slate-50/50"
                              : "",
                          ].join(" ")}
                        >
                          <td className="py-2 pl-4 pr-2 font-mono text-slate-400">
                            {idx + 1}
                          </td>
                          <td className="py-2 pr-3">
                            <div className="font-mono font-semibold text-slate-900">
                              {attr.col}
                            </div>
                            <div className="flex items-center gap-1 mt-0.5">
                              <span
                                className={[
                                  "px-1 py-0 rounded text-[9px] font-semibold uppercase",
                                  attr.isCountry
                                    ? "bg-sky-100 text-sky-800"
                                    : attr.source === "global"
                                      ? "bg-slate-100 text-slate-600"
                                      : "bg-purple-100 text-purple-800",
                                ].join(" ")}
                              >
                                {attr.isCountry
                                  ? "country"
                                  : attr.source === "global"
                                    ? "global"
                                    : "dataset extra"}
                              </span>
                            </div>
                          </td>
                          <td className="py-2 pr-3 text-slate-700 leading-snug">
                            <span className="font-medium text-slate-800">
                              {attr.promptLabel}
                            </span>
                            <span className="text-slate-400 font-mono">:</span>
                          </td>
                          <td className="py-2 pr-3">
                            <div className="font-mono text-[11px] text-slate-700">
                              {attr.nPresent.toLocaleString()}/
                              {data.nRespondents.toLocaleString()}
                            </div>
                            <div className="w-24 h-1.5 rounded-full bg-slate-100 overflow-hidden mt-1">
                              <div
                                className={
                                  attr.nMissing === 0
                                    ? "h-full bg-emerald-500"
                                    : "h-full bg-amber-400"
                                }
                                style={{ width: `${attr.pctPresent}%` }}
                              />
                            </div>
                            {attr.nMissing > 0 && (
                              <div className="text-[10px] text-amber-700 mt-0.5">
                                {attr.nMissing.toLocaleString()} NA
                              </div>
                            )}
                          </td>
                          <td className="py-2 pr-4">
                            <div className="flex flex-wrap gap-1">
                              {attr.levels.map((lv) => {
                                const isActive =
                                  attrCol === attr.col && attrVal === lv.value;
                                return (
                                  <button
                                    key={lv.value}
                                    type="button"
                                    onClick={() =>
                                      handleClickLevelChip(attr.col, lv.value)
                                    }
                                    title={`Filter respondents where ${attr.col} = ${lv.value} (${lv.count} / ${lv.pct}%)`}
                                    className={[
                                      "inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[11px] transition-colors cursor-pointer",
                                      isActive
                                        ? "bg-blue-600 text-white border-blue-600 font-medium"
                                        : "bg-slate-50 text-slate-700 border-slate-200 hover:bg-blue-50 hover:border-blue-300",
                                    ].join(" ")}
                                  >
                                    <span>{lv.value}</span>
                                    <span
                                      className={[
                                        "font-mono text-[10px]",
                                        isActive
                                          ? "text-blue-100"
                                          : "text-slate-400",
                                      ].join(" ")}
                                    >
                                      {lv.count.toLocaleString()} ({lv.pct}%)
                                    </span>
                                  </button>
                                );
                              })}
                              {attr.nMissing > 0 && (
                                <button
                                  type="button"
                                  onClick={() =>
                                    handleClickLevelChip(
                                      attr.col,
                                      "__MISSING__"
                                    )
                                  }
                                  className={[
                                    "inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[11px] transition-colors cursor-pointer",
                                    attrCol === attr.col &&
                                    attrVal === "__MISSING__"
                                      ? "bg-amber-600 text-white border-amber-600 font-medium"
                                      : "bg-amber-50 text-amber-800 border-amber-200 hover:bg-amber-100",
                                  ].join(" ")}
                                >
                                  <span>Missing (NA)</span>
                                  <span className="font-mono text-[10px]">
                                    {attr.nMissing.toLocaleString()}
                                  </span>
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
}
