"use client";

import { ColumnDef, SortingState, flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from "@tanstack/react-table";
import { ArrowDown, ArrowUp } from "lucide-react";
import { useState } from "react";
import { Label } from "@/components/ui/label";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatNumber } from "@/lib/format";

type Cell = number | string | null;

// val_history.csv as a sortable table; the best step is highlighted; validator column groups can be hidden
export default function ValidationTable({
  columns,
  rows,
  validators,
  bestStep,
}: {
  columns: string[];
  rows: Cell[][];
  validators: string[];
  bestStep: number | null;
}) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [shown, setShown] = useState<string[]>(validators);
  const stepIndex = columns.indexOf("step");
  if (stepIndex === -1) {
    throw new Error(`val_history.csv has no "step" column (columns: ${columns.join(", ")})`);
  }

  const visible: { column: string; index: number }[] = [];
  columns.forEach((column, index) => {
    const slash = column.indexOf("/");
    if (slash === -1 || shown.includes(column.slice(0, slash))) {
      visible.push({ column, index });
    }
  });

  const tableColumns: ColumnDef<Cell[]>[] = visible.map(({ column, index }) => ({
    id: column,
    accessorFn: (row) => (row[index] === null ? undefined : row[index]),
    sortUndefined: "last",
    header: ({ column: tableColumn }) => {
      const sorted = tableColumn.getIsSorted();
      return (
        <button
          type="button"
          className="inline-flex items-center gap-1 font-mono text-[11px] font-medium hover:text-foreground"
          onClick={() => tableColumn.toggleSorting(sorted === "asc")}
        >
          {column}
          {sorted === "asc" ? <ArrowUp className="size-3" /> : sorted === "desc" ? <ArrowDown className="size-3" /> : null}
        </button>
      );
    },
    cell: ({ row }) => {
      const value = row.original[index];
      if (typeof value === "number") {
        return <span className="tabular-nums">{formatNumber(value)}</span>;
      }
      return value;
    },
  }));

  const table = useReactTable({
    data: rows,
    columns: tableColumns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Label className="text-xs text-muted-foreground">validators</Label>
        <ToggleGroup type="multiple" variant="outline" size="sm" value={shown} onValueChange={setShown}>
          {validators.map((validator) => (
            <ToggleGroupItem key={validator} value={validator} className="px-2.5 font-mono text-xs">
              {validator}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <span className="text-xs text-muted-foreground">
          Highlighted: best step {bestStep === null ? "—" : bestStep}. Empty cells: that validator did not run at that step. Click a header to sort.
        </span>
      </div>
      <div className="max-h-[70vh] overflow-auto rounded-xl ring-1 ring-foreground/10">
        <Table>
          <TableHeader className="sticky top-0 z-10 bg-card">
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id} className="text-right first:text-left">
                    {flexRender(header.column.columnDef.header, header.getContext())}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => {
              const isBest = bestStep !== null && row.original[stepIndex] === bestStep;
              return (
                <TableRow key={row.id} className={isBest ? "bg-amber-100 hover:bg-amber-100 dark:bg-amber-400/15 dark:hover:bg-amber-400/15" : undefined}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id} className="text-right text-xs first:text-left">
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
