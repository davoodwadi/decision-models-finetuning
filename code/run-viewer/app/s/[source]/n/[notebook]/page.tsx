import { CircleAlert } from "lucide-react";
import AutoRefresh from "@/components/AutoRefresh";
import PageHeader from "@/components/page-header";
import RunsTable, { RunTableRow } from "@/components/runs-table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { listRuns } from "@/lib/runs";
import { formatAge, hyperparameterColumns, mainHyperparameters, sourceHref } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function NotebookPage({ params }: { params: Promise<{ source: string; notebook: string }> }) {
  const raw = await params;
  const source = decodeURIComponent(raw.source);
  const notebook = decodeURIComponent(raw.notebook);
  const { runs, otherFolders } = listRuns(source, notebook);
  const now = Date.now();
  const rows: RunTableRow[] = [];
  const errors: { name: string; error: string }[] = [];
  for (const run of runs) {
    if ("error" in run) {
      errors.push(run);
      continue;
    }
    const s = run.summary;
    rows.push({
      name: s.name,
      status: s.status,
      stepsDone: s.stepsDone,
      totalSteps: s.totalSteps,
      bestMetric: s.bestMetric,
      bestMode: s.bestMode,
      bestStep: s.bestStep,
      bestValue: s.bestValue,
      bestFromSummary: s.bestFromSummary,
      updated: s.updated,
      age: formatAge(s.updated, now),
      hyperparameters: mainHyperparameters(s.config),
    });
  }
  // A hyperparameter gets a column when any run of this notebook has it
  const columns = hyperparameterColumns(rows);
  const anyRunning = rows.some((row) => row.status === "running");

  return (
    <>
      <PageHeader crumbs={[{ label: "runs", href: "/" }, { label: source, href: sourceHref(source) }, { label: notebook }]} title={notebook}>
        <span className="text-muted-foreground">
          {rows.length} readable run{rows.length === 1 ? "" : "s"}
          {errors.length > 0 ? `, ${errors.length} with errors` : ""}
        </span>
        <AutoRefresh active={anyRunning} seconds={10} />
      </PageHeader>
      {rows.length > 0 ? <RunsTable source={source} notebook={notebook} rows={rows} hyperparameterColumns={columns} /> : null}
      {errors.map((run) => (
        <Alert key={run.name} variant="destructive">
          <CircleAlert />
          <AlertTitle className="font-mono">{run.name}</AlertTitle>
          <AlertDescription className="font-mono text-xs break-all">{run.error}</AlertDescription>
        </Alert>
      ))}
      {otherFolders.length > 0 ? (
        <p className="text-xs text-muted-foreground">Folders without run.json (not shown): {otherFolders.join(", ")}</p>
      ) : null}
    </>
  );
}
