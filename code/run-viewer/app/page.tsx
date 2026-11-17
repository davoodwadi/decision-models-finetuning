import Link from "next/link";
import LocalTime from "@/components/LocalTime";
import { listNotebooks, runsRoot } from "@/lib/runs";
import { formatAge, notebookHref } from "@/lib/format";

export const dynamic = "force-dynamic";

export default function Home() {
  const root = runsRoot();
  const notebooks = listNotebooks();
  const now = Date.now();
  return (
    <div>
      <h1>Notebooks</h1>
      <p className="crumbs">{root}</p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>notebook</th>
              <th>runs</th>
              <th>newest update</th>
              <th>other folders</th>
            </tr>
          </thead>
          <tbody>
            {notebooks.map((notebook) => (
              <tr key={notebook.name}>
                <td>
                  <Link href={notebookHref(notebook.name)}>{notebook.name}</Link>
                </td>
                <td className="num">{notebook.runCount}</td>
                <td>
                  <LocalTime ms={notebook.newestUpdate} /> <span className="muted small">{formatAge(notebook.newestUpdate, now)}</span>
                </td>
                <td className="muted small">{notebook.otherFolders.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
