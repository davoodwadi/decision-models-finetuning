import Link from "next/link";
import LocalTime from "@/components/LocalTime";
import AutoRefresh from "@/components/AutoRefresh";
import MetricChart, { Series } from "@/components/MetricChart";
import RollingChart, { StepGroup } from "@/components/RollingChart";
import StatusBadge from "@/components/StatusBadge";
import { Table, readRun, validatorMetrics } from "@/lib/runs";
import { columnIndex, numberAt } from "@/lib/csv";
import { formatNumber, notebookHref } from "@/lib/format";

export const dynamic = "force-dynamic";

const LOSS_COLUMNS = ["total_loss", "kl_loss", "raw_beta_loss", "weighted_beta_loss"];

function metricSeries(val: Table, validators: string[], metric: string, file: string): Series[] {
  const stepIndex = columnIndex(val, "step", file);
  const series: Series[] = [];
  for (const validator of validators) {
    const column = `${validator}/${metric}`;
    const index = val.columns.indexOf(column);
    if (index === -1) {
      continue;
    }
    const points = [];
    for (const row of val.rows) {
      const value = numberAt(row, index, column, file);
      if (value === null) {
        // NaN: this validator did not run at this step
        continue;
      }
      const step = numberAt(row, stepIndex, "step", file);
      if (step === null) {
        throw new Error(`${file}: a row has no step`);
      }
      points.push({ x: step, y: value });
    }
    series.push({ name: validator, points });
  }
  return series;
}

function groupStepsByDataset(steps: Table, columns: string[], file: string): StepGroup[] {
  const stepIndex = columnIndex(steps, "step", file);
  const datasetIndex = columnIndex(steps, "dataset", file);
  const valueIndexes = columns.map((column) => columnIndex(steps, column, file));
  const groups = new Map<string, StepGroup>();
  for (const row of steps.rows) {
    const dataset = row[datasetIndex];
    if (typeof dataset !== "string") {
      throw new Error(`${file}: dataset column holds ${JSON.stringify(dataset)}`);
    }
    const step = numberAt(row, stepIndex, "step", file);
    if (step === null) {
      throw new Error(`${file}: a row has no step`);
    }
    let group = groups.get(dataset);
    if (group === undefined) {
      group = { name: dataset, steps: [], values: {} };
      for (const column of columns) {
        group.values[column] = [];
      }
      groups.set(dataset, group);
    }
    group.steps.push(step);
    for (let i = 0; i < columns.length; i++) {
      group.values[columns[i]].push(numberAt(row, valueIndexes[i], columns[i], file));
    }
  }
  return [...groups.values()].sort((a, b) => a.name.localeCompare(b.name));
}

function cellText(value: number | string | null): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  return value === null ? "" : value;
}

export default async function RunPage({ params }: { params: Promise<{ notebook: string; run: string }> }) {
  const raw = await params;
  const notebook = decodeURIComponent(raw.notebook);
  const run = decodeURIComponent(raw.run);
  const detail = readRun(notebook, run);
  const { summary, val, steps } = detail;
  const valFile = `${notebook}/${run}/val_history.csv`;
  const stepsFile = `${notebook}/${run}/steps.csv`;

  const { validators, metrics } = validatorMetrics(val.columns);
  const stepIndex = val.rows.length > 0 ? columnIndex(val, "step", valFile) : -1;

  const lossColumns = LOSS_COLUMNS.filter((column) => steps.columns.includes(column));
  const splitColumns = steps.columns.filter((column) => column.startsWith("grad_") && column !== "grad_norm");
  const lossGroups = steps.rows.length > 0 ? groupStepsByDataset(steps, lossColumns, stepsFile) : [];
  const gradColumns = splitColumns.length > 0 ? [...splitColumns, "grad_norm"] : [];
  const gradGroups = gradColumns.length > 0 && steps.rows.length > 0 ? groupStepsByDataset(steps, gradColumns, stepsFile) : [];

  return (
    <div>
      <p className="crumbs">
        <Link href="/">notebooks</Link> / <Link href={notebookHref(notebook)}>{notebook}</Link> /
      </p>
      <h1>{run}</h1>
      <p>
        <StatusBadge status={summary.status} /> · step {summary.stepsDone} / {summary.totalSteps === null ? "?" : summary.totalSteps} ·
        best {summary.bestMetric} ({summary.bestMode}) = {formatNumber(summary.bestValue)} at step{" "}
        {summary.bestStep === null ? "—" : summary.bestStep}
        {summary.bestFromSummary || summary.bestValue === null ? "" : " (so far)"} · created {summary.created} · updated{" "}
        <LocalTime ms={summary.updated} /> · resumes {summary.resumes.length} · <AutoRefresh active={summary.status === "running"} seconds={10} />
      </p>
      {steps.droppedPartialLine || val.droppedPartialLine ? (
        <p className="muted small">A partially written last line was skipped (the run is still writing).</p>
      ) : null}

      <details className="card">
        <summary>config</summary>
        <dl className="config">
          {summary.model === null ? null : (
            <>
              <dt>model</dt>
              <dd>{JSON.stringify(summary.model)}</dd>
            </>
          )}
          {Object.entries(summary.config).map(([key, value]) => (
            <div key={key} style={{ display: "contents" }}>
              <dt>{key}</dt>
              <dd>{JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      </details>

      <h2>Validation metrics</h2>
      {metrics.length === 0 ? (
        <p className="muted">No validation rows yet.</p>
      ) : (
        <div className="grid">
          {metrics.map((metric) => (
            <MetricChart
              key={metric}
              title={metric}
              series={metricSeries(val, validators, metric, valFile)}
              xLabel="step"
              markX={summary.bestStep}
            />
          ))}
        </div>
      )}

      <h2>Training loss by dataset</h2>
      {lossGroups.length === 0 ? (
        <p className="muted">No training steps yet.</p>
      ) : (
        <RollingChart title="training" columns={lossColumns} groups={lossGroups} defaultWindow={25} />
      )}

      {gradGroups.length > 0 ? (
        <>
          <h2>Gradient split by dataset</h2>
          <RollingChart title="gradient" columns={gradColumns} groups={gradGroups} defaultWindow={25} />
        </>
      ) : null}

      <h2>Validation table</h2>
      <p className="muted small">Highlighted: best step {summary.bestStep === null ? "—" : summary.bestStep}. Empty cells: validator did not run at that step.</p>
      {val.rows.length === 0 ? (
        <p className="muted">No validation rows yet.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {val.columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {val.rows.map((row, rowNumber) => (
                <tr key={rowNumber} className={row[stepIndex] === summary.bestStep ? "best" : undefined}>
                  {row.map((value, i) => (
                    <td key={i} className={typeof value === "number" ? "num" : undefined}>
                      {cellText(value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
