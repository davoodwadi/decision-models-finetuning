import { AuditWorkspace } from "@/components/AuditWorkspace";
import { discoverDatasets, getDatasetPayload } from "@/lib/store";
import type {
  AuditFlag,
  DecisionFilter,
  InitialViewState,
  ReviewFilter,
  WorkspaceTab,
} from "@/lib/types";

export const dynamic = "force-dynamic";

const VALID_REVIEW_FILTERS = new Set<ReviewFilter>(["all", "unreviewed", "reviewed"]);
const VALID_DECISION_FILTERS = new Set<DecisionFilter>(["all", "included", "excluded"]);
const VALID_FLAGS = new Set<AuditFlag>([
  "blocking",
  "suspect",
  "unsaved",
  "overridden",
  "tagged",
  "option_relabeled",
  "codebook_diff",
  "text_changed",
  "options_differ",
  "not_in_source",
  "identical",
]);

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const { defaultDs, datasets } = await discoverDatasets();

  const dsParam = typeof params.ds === "string" ? params.ds : undefined;
  const activeDs =
    dsParam && datasets.some((d) => d.id === dsParam) ? dsParam : defaultDs;

  const initialPayload = await getDatasetPayload(activeDs);

  const rawFamily = params.family;
  const families =
    typeof rawFamily === "string"
      ? [rawFamily]
      : Array.isArray(rawFamily)
        ? rawFamily.filter(Boolean)
        : undefined;
  const rawCategory = params.cat;
  const categories =
    typeof rawCategory === "string"
      ? [rawCategory]
      : Array.isArray(rawCategory)
        ? rawCategory.filter(Boolean)
        : undefined;
  const rawTag = params.tag;
  const tags =
    typeof rawTag === "string"
      ? [rawTag]
      : Array.isArray(rawTag)
        ? rawTag.filter(Boolean)
        : undefined;

  const tabParam = typeof params.tab === "string" ? params.tab : undefined;
  const tab: WorkspaceTab | undefined =
    tabParam === "personas" || tabParam === "questions" || tabParam === "source"
      ? tabParam
      : undefined;

  // Parse unified params (rev, dec, flag) and translate legacy (mode, src, srcrev) params
  const revParam = typeof params.rev === "string" ? params.rev : undefined;
  let reviewFilter: ReviewFilter | undefined =
    revParam && VALID_REVIEW_FILTERS.has(revParam as ReviewFilter)
      ? (revParam as ReviewFilter)
      : undefined;

  const decParam = typeof params.dec === "string" ? params.dec : undefined;
  let decisionFilter: DecisionFilter | undefined =
    decParam && VALID_DECISION_FILTERS.has(decParam as DecisionFilter)
      ? (decParam as DecisionFilter)
      : undefined;

  const rawFlag = params.flag;
  const rawFlagList =
    typeof rawFlag === "string"
      ? [rawFlag]
      : Array.isArray(rawFlag)
        ? rawFlag
        : [];
  const flagSet = new Set<AuditFlag>();
  for (const f of rawFlagList) {
    if (VALID_FLAGS.has(f as AuditFlag)) {
      flagSet.add(f as AuditFlag);
    }
  }

  let skip: boolean | undefined = params.skip === "1" ? true : undefined;

  const legacyMode = typeof params.mode === "string" ? params.mode : undefined;
  if (legacyMode) {
    if (legacyMode === "blocking") flagSet.add("blocking");
    else if (legacyMode === "suspects") flagSet.add("suspect");
    else if (legacyMode === "unsaved") flagSet.add("unsaved");
    else if (legacyMode === "overridden") flagSet.add("overridden");
    else if (legacyMode === "tagged") flagSet.add("tagged");
    else if (legacyMode === "unreviewed" && !reviewFilter) reviewFilter = "unreviewed";
    else if (legacyMode === "included" && !decisionFilter) decisionFilter = "included";
    else if (legacyMode === "excluded" && !decisionFilter) decisionFilter = "excluded";
    else if (legacyMode === "all" && skip === undefined) skip = true;
  }

  const legacySrc = typeof params.src === "string" ? params.src : undefined;
  if (legacySrc) {
    if (legacySrc === "needs_check") flagSet.add("codebook_diff");
    else if (legacySrc === "flagged") flagSet.add("suspect");
    else if (VALID_FLAGS.has(legacySrc as AuditFlag)) flagSet.add(legacySrc as AuditFlag);
  }

  const legacySrcRev = typeof params.srcrev === "string" ? params.srcrev : undefined;
  if (legacySrcRev && !reviewFilter) {
    if (legacySrcRev === "unchecked") reviewFilter = "unreviewed";
    else if (legacySrcRev === "checked") reviewFilter = "reviewed";
  }

  const hasExplicitFilterParam =
    revParam !== undefined ||
    decParam !== undefined ||
    rawFlag !== undefined ||
    legacyMode !== undefined ||
    legacySrc !== undefined ||
    legacySrcRev !== undefined;

  const initialView: InitialViewState = {
    hasParams: Object.keys(params).length > 0,
    tab,
    rowKey: typeof params.row === "string" ? params.row : undefined,
    reviewFilter,
    decisionFilter,
    flags: hasExplicitFilterParam ? Array.from(flagSet) : undefined,
    skip,
    study: typeof params.study === "string" ? params.study : undefined,
    families,
    categories,
    tags,
    q: typeof params.q === "string" ? params.q : undefined,
  };

  return (
    <AuditWorkspace
      initialDatasets={datasets}
      initialPayload={initialPayload}
      initialView={initialView}
    />
  );
}
