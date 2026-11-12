import { AuditWorkspace } from "@/components/AuditWorkspace";
import { discoverDatasets, getDatasetPayload } from "@/lib/store";
import type { InitialViewState, TriageMode } from "@/lib/types";

export const dynamic = "force-dynamic";

const VALID_MODES = new Set<TriageMode>([
  "suspects",
  "unsaved",
  "overridden",
  "tagged",
  "unreviewed",
  "substantive",
  "included",
  "excluded",
  "all",
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

  const modeParam = typeof params.mode === "string" ? params.mode : undefined;
  const rawFamily = params.family;
  const families =
    typeof rawFamily === "string"
      ? [rawFamily]
      : Array.isArray(rawFamily)
        ? rawFamily.filter(Boolean)
        : undefined;
  const rawTag = params.tag;
  const tags =
    typeof rawTag === "string"
      ? [rawTag]
      : Array.isArray(rawTag)
        ? rawTag.filter(Boolean)
        : undefined;

  const initialView: InitialViewState = {
    hasParams: Object.keys(params).length > 0,
    rowKey: typeof params.row === "string" ? params.row : undefined,
    mode:
      modeParam && VALID_MODES.has(modeParam as TriageMode)
        ? (modeParam as TriageMode)
        : undefined,
    skip: params.skip === "1" ? true : undefined,
    study: typeof params.study === "string" ? params.study : undefined,
    families,
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
