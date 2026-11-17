import Link from "next/link";
import LocalTime from "@/components/LocalTime";
import PageHeader from "@/components/page-header";
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { listNotebooks, runsRoot } from "@/lib/runs";
import { formatAge, notebookHref } from "@/lib/format";

export const dynamic = "force-dynamic";

export default function Home() {
  const root = runsRoot();
  const notebooks = listNotebooks();
  const now = Date.now();
  return (
    <>
      <PageHeader crumbs={[{ label: "notebooks" }]} title="Notebooks">
        <span className="font-mono text-xs break-all text-muted-foreground">{root}</span>
      </PageHeader>
      <Card className="py-0">
        <CardContent className="px-0">
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
                    <Link href={notebookHref(notebook.name)} className="hover:underline">
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
        </CardContent>
      </Card>
    </>
  );
}
