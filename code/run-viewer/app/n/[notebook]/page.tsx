import Link from "next/link";
import LocalTime from "@/components/LocalTime";
import AutoRefresh from "@/components/AutoRefresh";
import StatusBadge from "@/components/StatusBadge";
import { listRuns } from "@/lib/runs";
import { formatAge, formatNumber, runHref } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function NotebookPage({ params }: { params: Promise<{ notebook: string }> }) {
  const notebook = decodeURIComponent((await params).notebook);
  const { runs, otherFolders } = listRuns(notebook);
  const now = Date.now();
  let anyRunning = false;
  for (const run of runs) {
    if ("summary" in run && run.summary.status === "running") {
      anyRunning = true;
    }
  }
  return (
    <div>
      <p className="crumbs">
        <Link href="/">notebooks</Link> /
      </p>
      <h1>{notebook}</h1>
      <p>
        <AutoRefresh active={anyRunning} seconds={10} />
      </p>
      <form action={`/n/${encodeURIComponent(notebook)}/compare`} method="get">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>compare</th>
                <th>run</th>
                <th>status</th>
                <th>steps done / total</th>
                <th>best metric</th>
                <th>best step</th>
                <th>best value</th>
                <th>updated</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) =>
                "error" in run ? (
                  <tr key={run.name}>
                    <td></td>
                    <td>{run.name}</td>
                    <td colSpan={6}>
                      <div className="error">{run.error}</div>
                    </td>
                  </tr>
                ) : (
                  <tr key={run.name}>
                    <td>
                      <input type="checkbox" name="run" value={run.name} />
                    </td>
                    <td>
                      <Link href={runHref(notebook, run.name)}>{run.name}</Link>
                    </td>
                    <td>
                      <StatusBadge status={run.summary.status} />
                    </td>
                    <td className="num">
                      {run.summary.stepsDone} / {run.summary.totalSteps === null ? "?" : run.summary.totalSteps}
                    </td>
                    <td>
                      {run.summary.bestMetric} <span className="muted small">({run.summary.bestMode})</span>
                    </td>
                    <td className="num">{run.summary.bestStep === null ? "—" : run.summary.bestStep}</td>
                    <td className="num">
                      {formatNumber(run.summary.bestValue)}
                      {run.summary.bestFromSummary || run.summary.bestValue === null ? "" : <span className="muted small"> so far</span>}
                    </td>
                    <td>
                      <LocalTime ms={run.summary.updated} /> <span className="muted small">{formatAge(run.summary.updated, now)}</span>
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
        <p>
          <button type="submit">Compare selected runs</button>{" "}
          <span className="muted small">total steps are known only once a run has finished (from run.json summary)</span>
        </p>
      </form>
      {otherFolders.length > 0 ? <p className="muted small">Folders without run.json (not shown): {otherFolders.join(", ")}</p> : null}
    </div>
  );
}
