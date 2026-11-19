import Link from "next/link";
import LocalTime from "@/components/LocalTime";
import PageHeader from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listNotebooks, runSources } from "@/lib/runs";
import { formatAge, notebookHref } from "@/lib/format";

export const dynamic = "force-dynamic";

export default function Home() {
  const sources = runSources();
  const now = Date.now();
  return (
    <>
      <PageHeader crumbs={[{ label: "runs" }]} title="Runs">
        <span className="text-muted-foreground">
          {sources.length} run folder{sources.length === 1 ? "" : "s"}
        </span>
      </PageHeader>
      {sources.map((source) => {
        const notebooks = listNotebooks(source.name);
        return (
          <Card key={source.name} id={`source-${source.name}`} className="scroll-mt-4 gap-0 pb-0">
            <CardHeader className="pb-3">
              <CardTitle className="font-mono">{source.name}</CardTitle>
              <span className="font-mono text-xs break-all text-muted-foreground">{source.root}</span>
            </CardHeader>
            <CardContent className="px-0">
              {notebooks.length === 0 ? (
                <p className="px-6 pb-6 text-sm text-muted-foreground">No notebook folders yet.</p>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="pl-4">notebook</TableHead>
                      <TableHead className="text-right">runs</TableHead>
                      <TableHead>newest update</TableHead>
                      <TableHead>other folders</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {notebooks.map((notebook) => (
                      <TableRow key={notebook.name}>
                        <TableCell className="pl-4 font-mono">
                          <Link href={notebookHref(source.name, notebook.name)} className="hover:underline">
                            {notebook.name}
                          </Link>
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{notebook.runCount}</TableCell>
                        <TableCell>
                          <LocalTime ms={notebook.newestUpdate} />{" "}
                          <span className="text-xs text-muted-foreground">{formatAge(notebook.newestUpdate, now)}</span>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">{notebook.otherFolders.join(", ")}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        );
      })}
    </>
  );
}
