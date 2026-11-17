import Link from "next/link";
import AutoRefresh from "@/components/AutoRefresh";
import MetricChart, { Point, Series } from "@/components/MetricChart";
import StatusBadge from "@/components/StatusBadge";
import { RunDetail, readRun } from "@/lib/runs";
import { columnIndex, numberAt } from "@/lib/csv";
import { formatNumber, notebookHref, runHref } from "@/lib/format";

export const dynamic = "force-dynamic";

function metricPoints(detail: RunDetail, column: string): Point[] {
  const val = detail.val;
  const file = `${detail.notebook}/${detail.summary.name}/val_history.csv`;
  const index = val.columns.indexOf(column);
  if (index === -1 || val.rows.length === 0) {
    return [];
  }
  const stepIndex = columnIndex(val, "step", file);
  const points: Point[] = [];
  for (const row of val.rows) {
    const value = numberAt(row, index, column, file);
    if (value === null) {
      continue;
    }
    const step = numberAt(row, stepIndex, "step", file);
    if (step === null) {
      throw new Error(`${file}: a row has no step`);
    }
    points.push({ x: step, y: value });
  }
  return points;
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
  const runs = query.run === undefined ? [] : Array.isArray(query.run) ? query.run : [query.run];
  if (runs.length === 0) {
    return (
      <div>
        <p className="crumbs">
          <Link href="/">notebooks</Link> / <Link href={notebookHref(notebook)}>{notebook}</Link> /
        </p>
        <h1>Compare</h1>
        <p>No runs selected. Tick runs on the notebook page and press Compare.</p>
      </div>
    );
  }
  if (Array.isArray(query.metric)) {
    throw new Error(`Only one metric can be compared at a time, got ${query.metric.join(", ")}`);
  }
  const details = runs.map((run) => readRun(notebook, run));

  const metricOptions: string[] = [];
  for (const detail of details) {
    for (const column of detail.val.columns) {
      if (column.includes("/") && !metricOptions.includes(column)) {
        metricOptions.push(column);
      }
    }
  }
  const metric = query.metric === undefined ? details[0].summary.bestMetric : query.metric;

  const series: Series[] = details.map((detail) => ({ name: detail.summary.name, points: metricPoints(detail, metric) }));
  let anyRunning = false;
  for (const detail of details) {
    if (detail.summary.status === "running") {
      anyRunning = true;
    }
  }

  return (
    <div>
      <p className="crumbs">
        <Link href="/">notebooks</Link> / <Link href={notebookHref(notebook)}>{notebook}</Link> /
      </p>
      <h1>Compare {details.length} runs</h1>
      <p>
        <AutoRefresh active={anyRunning} seconds={10} />
      </p>
      <form method="get" className="controls">
        {runs.map((run) => (
          <input key={run} type="hidden" name="run" value={run} />
        ))}
        <label>
          metric{" "}
          <select name="metric" defaultValue={metric}>
            {metricOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <button type="submit">Show</button>
      </form>
      <MetricChart title={metric} series={series} xLabel="step" height={420} />
      <h2>Runs</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>run</th>
              <th>status</th>
              <th>points</th>
              <th>min</th>
              <th>max</th>
              <th>last</th>
            </tr>
          </thead>
          <tbody>
            {series.map((s, i) => {
              const values = s.points.map((p) => p.y);
              const detail = details[i];
              return (
                <tr key={s.name}>
                  <td>
                    <Link href={runHref(notebook, s.name)}>{s.name}</Link>
                  </td>
                  <td>
                    <StatusBadge status={detail.summary.status} />
                  </td>
                  <td className="num">
                    {values.length}
                    {detail.val.columns.includes(metric) ? "" : <span className="muted small"> (column absent)</span>}
                  </td>
                  <td className="num">{values.length === 0 ? "" : formatNumber(Math.min(...values))}</td>
                  <td className="num">{values.length === 0 ? "" : formatNumber(Math.max(...values))}</td>
                  <td className="num">{values.length === 0 ? "" : formatNumber(values[values.length - 1])}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
