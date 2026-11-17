import AutoRefresh from "@/components/AutoRefresh";
import LocalTime from "@/components/LocalTime";
import LineChart, { ChartSeries, XY } from "@/components/charts/line-chart";
import RollingChart, { StepGroup } from "@/components/charts/rolling-chart";
import PageHeader from "@/components/page-header";
import StatusBadge from "@/components/status-badge";
import ValidationTable from "@/components/validation-table";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Table as UiTable, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Table, readRun, validatorMetrics } from "@/lib/runs";
import { columnIndex, numberAt } from "@/lib/csv";
import { formatNumber, notebookHref } from "@/lib/format";

export const dynamic = "force-dynamic";

const LOSS_COLUMNS = ["total_loss", "kl_loss", "raw_beta_loss", "weighted_beta_loss"];

function stepOf(table: Table, row: (number | string | null)[], stepIndex: number, file: string): number {
  const step = numberAt(row, stepIndex, "step", file);
  if (step === null) {
    throw new Error(`${file}: a row has no step`);
  }
  return step;
}

function metricSeries(val: Table, validators: string[], metric: string, file: string): ChartSeries[] {
  const stepIndex = columnIndex(val, "step", file);
  const series: ChartSeries[] = [];
  validators.forEach((validator, slot) => {
    const column = `${validator}/${metric}`;
    const index = val.columns.indexOf(column);
    if (index === -1) {
      return;
    }
    const points: XY[] = [];
    for (const row of val.rows) {
      const value = numberAt(row, index, column, file);
      if (value === null) {
        // NaN: this validator did not run at this step
        continue;
      }
      points.push([stepOf(val, row, stepIndex, file), value]);
    }
    series.push({ name: validator, slot, points });
  });
  return series;
}

function stepColumnSeries(steps: Table, column: string, file: string): XY[] {
  const stepIndex = columnIndex(steps, "step", file);
  const index = columnIndex(steps, column, file);
  const points: XY[] = [];
  for (const row of steps.rows) {
    const value = numberAt(row, index, column, file);
    if (value !== null) {
      points.push([stepOf(steps, row, stepIndex, file), value]);
    }
  }
  return points;
}

function groupStepsByDataset(steps: Table, columns: string[], file: string): StepGroup[] {
  const stepIndex = columnIndex(steps, "step", file);
  const datasetIndex = columnIndex(steps, "dataset", file);
  const valueIndexes = columns.map((column) => columnIndex(steps, column, file));
  const groups = new Map<string, Omit<StepGroup, "slot">>();
  for (const row of steps.rows) {
    const dataset = row[datasetIndex];
    if (typeof dataset !== "string") {
      throw new Error(`${file}: dataset column holds ${JSON.stringify(dataset)}`);
    }
    let group = groups.get(dataset);
    if (group === undefined) {
      group = { name: dataset, steps: [], values: {} };
      for (const column of columns) {
        group.values[column] = [];
      }
      groups.set(dataset, group);
    }
    group.steps.push(stepOf(steps, row, stepIndex, file));
    for (let i = 0; i < columns.length; i++) {
      group.values[columns[i]].push(numberAt(row, valueIndexes[i], columns[i], file));
    }
  }
  // Datasets sorted by name: each dataset keeps the same color slot on every chart
  const sorted = [...groups.values()].sort((a, b) => a.name.localeCompare(b.name));
  return sorted.map((group, slot) => ({ ...group, slot }));
}

function Stat({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="text-sm font-medium tabular-nums">{children}</span>
    </div>
  );
}

export default async function RunPage({ params }: { params: Promise<{ notebook: string; run: string }> }) {
  const raw = await params;
  const notebook = decodeURIComponent(raw.notebook);
  const run = decodeURIComponent(raw.run);
  const { summary, val, steps } = readRun(notebook, run);
  const valFile = `${notebook}/${run}/val_history.csv`;
  const stepsFile = `${notebook}/${run}/steps.csv`;

  const { validators, metrics } = validatorMetrics(val.columns);
  const hasSteps = steps.rows.length > 0;
  const lossColumns = LOSS_COLUMNS.filter((column) => steps.columns.includes(column));
  const lossGroups = hasSteps ? groupStepsByDataset(steps, lossColumns, stepsFile) : [];
  const splitColumns = steps.columns.filter((column) => column.startsWith("grad_") && column !== "grad_norm");
  const gradColumns = splitColumns.length > 0 ? [...splitColumns, "grad_norm"] : [];
  const gradGroups = hasSteps && gradColumns.length > 0 ? groupStepsByDataset(steps, gradColumns, stepsFile) : [];
  const lrPoints = hasSteps && steps.columns.includes("lr") ? stepColumnSeries(steps, "lr", stepsFile) : [];
  const chartGroup = `val:${notebook}/${run}`;

  return (
    <>
      <PageHeader crumbs={[{ label: "notebooks", href: "/" }, { label: notebook, href: notebookHref(notebook) }, { label: run }]} title={run}>
        <StatusBadge status={summary.status} />
        <AutoRefresh active={summary.status === "running"} seconds={10} />
      </PageHeader>

      <Card size="sm">
        <CardContent className="flex flex-wrap gap-x-8 gap-y-3">
          <Stat label="steps done / total">
            {summary.stepsDone} / {summary.totalSteps === null ? "?" : summary.totalSteps}
          </Stat>
          <Stat label={`best ${summary.bestMetric} (${summary.bestMode})${summary.bestFromSummary ? "" : " so far"}`}>
            {formatNumber(summary.bestValue)}
          </Stat>
          <Stat label="best step">{summary.bestStep === null ? "—" : summary.bestStep}</Stat>
          <Separator orientation="vertical" className="hidden h-9 sm:block" />
          <Stat label="created">{summary.created.replace("T", " ")}</Stat>
          <Stat label="updated">
            <LocalTime ms={summary.updated} />
          </Stat>
          <Stat label="resumes">{summary.resumes.length}</Stat>
        </CardContent>
      </Card>
      {steps.droppedPartialLine || val.droppedPartialLine ? (
        <p className="text-xs text-muted-foreground">A partially written last line was skipped (the run is still writing).</p>
      ) : null}

      <Tabs defaultValue="validation" className="gap-4">
        <TabsList>
          <TabsTrigger value="validation">Validation</TabsTrigger>
          <TabsTrigger value="training">Training</TabsTrigger>
          {gradGroups.length > 0 ? <TabsTrigger value="gradients">Gradients</TabsTrigger> : null}
          <TabsTrigger value="table">Table</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>

        <TabsContent value="validation" className="flex flex-col gap-3">
          {metrics.length === 0 ? (
            <p className="text-sm text-muted-foreground">No validation rows yet.</p>
          ) : (
            <>
              <p className="text-xs text-muted-foreground">
                Charts are linked: hovering, zooming and hiding a validator in the legend apply to all of them. Drag to pan, ctrl + scroll to zoom,
                toolbar for box zoom, reset and PNG.
              </p>
              <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 2xl:grid-cols-3">
                {metrics.map((metric) => (
                  <LineChart
                    key={metric}
                    title={metric}
                    series={metricSeries(val, validators, metric, valFile)}
                    xName="step"
                    markX={summary.bestStep}
                    group={chartGroup}
                  />
                ))}
              </div>
            </>
          )}
        </TabsContent>

        <TabsContent value="training" className="flex flex-col gap-4">
          {lossGroups.length === 0 ? (
            <p className="text-sm text-muted-foreground">No training steps yet.</p>
          ) : (
            <>
              <RollingChart title="training loss" columns={lossColumns} groups={lossGroups} defaultWindow={25} />
              {lrPoints.length > 0 ? <LineChart title="lr" series={[{ name: "lr", slot: 0, points: lrPoints }]} xName="step" height={220} /> : null}
            </>
          )}
        </TabsContent>

        {gradGroups.length > 0 ? (
          <TabsContent value="gradients">
            <RollingChart title="gradient split" columns={gradColumns} groups={gradGroups} defaultWindow={25} />
          </TabsContent>
        ) : null}

        <TabsContent value="table">
          {val.rows.length === 0 ? (
            <p className="text-sm text-muted-foreground">No validation rows yet.</p>
          ) : (
            <ValidationTable columns={val.columns} rows={val.rows} validators={validators} bestStep={summary.bestStep} />
          )}
        </TabsContent>

        <TabsContent value="config">
          <Card size="sm" className="py-0">
            <CardContent className="px-0">
              <UiTable>
                <TableBody>
                  {summary.model === null ? null : (
                    <TableRow>
                      <TableCell className="w-56 pl-4 text-muted-foreground">model</TableCell>
                      <TableCell className="font-mono text-xs whitespace-pre-wrap">{JSON.stringify(summary.model)}</TableCell>
                    </TableRow>
                  )}
                  {Object.entries(summary.config).map(([key, value]) => (
                    <TableRow key={key}>
                      <TableCell className="w-56 pl-4 text-muted-foreground">{key}</TableCell>
                      <TableCell className="font-mono text-xs whitespace-pre-wrap">{JSON.stringify(value)}</TableCell>
                    </TableRow>
                  ))}
                  {summary.resumes.length > 0 ? (
                    <TableRow>
                      <TableCell className="w-56 pl-4 text-muted-foreground">resumes</TableCell>
                      <TableCell className="font-mono text-xs whitespace-pre-wrap">{JSON.stringify(summary.resumes, null, 1)}</TableCell>
                    </TableRow>
                  ) : null}
                </TableBody>
              </UiTable>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </>
  );
}
