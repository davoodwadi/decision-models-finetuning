import AutoRefresh from "@/components/AutoRefresh";
import CompareView, { CompareRun } from "@/components/compare-view";
import { XY } from "@/components/charts/line-chart";
import PageHeader from "@/components/page-header";
import { listRunNames, readRun, RunDetail } from "@/lib/runs";
import { columnIndex, numberAt } from "@/lib/csv";
import { MAX_SERIES, notebookHref } from "@/lib/format";

export const dynamic = "force-dynamic";

// Every <validator>/<metric> column of one run as (step, value) points; NaN cells (validator did not run) are skipped
function metricColumns(detail: RunDetail): Record<string, XY[]> {
  const val = detail.val;
  const file = `${detail.notebook}/${detail.summary.name}/val_history.csv`;
  const result: Record<string, XY[]> = {};
  if (val.rows.length === 0) {
    return result;
  }
  const stepIndex = columnIndex(val, "step", file);
  val.columns.forEach((column, index) => {
    if (!column.includes("/")) {
      return;
    }
    const points: XY[] = [];
    for (const row of val.rows) {
      const value = numberAt(row, index, column, file);
      if (value === null) {
        continue;
      }
      const step = numberAt(row, stepIndex, "step", file);
      if (step === null) {
        throw new Error(`${file}: a row has no step`);
      }
      points.push([step, value]);
    }
    result[column] = points;
  });
  return result;
}

export default async function ComparePage({
  params,
  searchParams,
}: {
  params: Promise<{ notebook: string }>;
  searchParams: Promise<{ run?: string | string[]; metric?: string | string[] }>;
}) {
  const notebook = decodeURIComponent((await params).notebook);
  const query = await searchParams;
  const requested = query.run === undefined ? [] : Array.isArray(query.run) ? query.run : [query.run];
  const crumbs = [{ label: "notebooks", href: "/" }, { label: notebook, href: notebookHref(notebook) }, { label: "compare" }];
  if (requested.length === 0) {
    return (
      <PageHeader crumbs={crumbs} title="Compare">
        <span className="text-muted-foreground">No runs selected. Tick runs on the notebook page and press Compare.</span>
      </PageHeader>
    );
  }
  if (requested.length > MAX_SERIES) {
    throw new Error(`At most ${MAX_SERIES} runs can be compared in one chart, got ${requested.length}`);
  }
  if (Array.isArray(query.metric)) {
    throw new Error(`Only one metric can be compared at a time, got ${query.metric.join(", ")}`);
  }

  // Runs in notebook order; color slot = position in the notebook when it fits the palette, else position in the selection
  const allNames = listRunNames(notebook);
  const selected = allNames.filter((name) => requested.includes(name));
  for (const name of requested) {
    if (!allNames.includes(name)) {
      throw new Error(`No run in ${notebook} named "${name}"`);
    }
  }
  const slotByNotebook = allNames.length <= MAX_SERIES;
  const runs: CompareRun[] = [];
  const metricOptions: string[] = [];
  let firstBestMetric = "";
  selected.forEach((name, position) => {
    const detail = readRun(notebook, name);
    if (firstBestMetric === "") {
      firstBestMetric = detail.summary.bestMetric;
    }
    const metrics = metricColumns(detail);
    for (const column of Object.keys(metrics)) {
      if (!metricOptions.includes(column)) {
        metricOptions.push(column);
      }
    }
    runs.push({
      name,
      slot: slotByNotebook ? allNames.indexOf(name) : position,
      status: detail.summary.status,
      bestStep: detail.summary.bestStep,
      metrics,
    });
  });
  const metric = query.metric === undefined ? firstBestMetric : query.metric;
  if (metricOptions.length > 0 && !metricOptions.includes(metric)) {
    throw new Error(`Metric "${metric}" is not a column of any selected run's val_history.csv`);
  }
  const anyRunning = runs.some((run) => run.status === "running");

  return (
    <>
      <PageHeader crumbs={crumbs} title={`Compare ${runs.length} run${runs.length === 1 ? "" : "s"}`}>
        <AutoRefresh active={anyRunning} seconds={10} />
      </PageHeader>
      {metricOptions.length === 0 ? (
        <p className="text-sm text-muted-foreground">The selected runs have no validation rows yet.</p>
      ) : (
        <CompareView notebook={notebook} runs={runs} metricOptions={metricOptions} initialMetric={metric} />
      )}
    </>
  );
}
