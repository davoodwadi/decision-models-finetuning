"use client";

import {
  ColumnDef,
  RowSelectionState,
  SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { ArrowDown, ArrowUp, ArrowUpDown, GitCompare } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import HyperparameterValue from "@/components/hyperparameter-value";
import LocalTime from "@/components/LocalTime";
import StatusBadge from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { MAX_SERIES, compareHref, formatNumber, runHref } from "@/lib/format";

export type RunTableRow = {
  name: string;
  status: string;
  stepsDone: number;
  totalSteps: number | null;
  bestMetric: string;
  bestMode: string;
  bestStep: number | null;
  bestValue: number | null;
  bestFromSummary: boolean;
  updated: number;
  age: string;
  // Main hyperparameters from run.json config, keyed by config key; a key is absent when the run's config lacks it
  hyperparameters: Record<string, string | number>;
};

function SortHeader({ label, column }: { label: string; column: { getIsSorted: () => false | "asc" | "desc"; toggleSorting: (desc?: boolean) => void } }) {
  const sorted = column.getIsSorted();
  return (
    <Button variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-xs" onClick={() => column.toggleSorting(sorted === "asc")}>
      {label}
      {sorted === "asc" ? <ArrowUp /> : sorted === "desc" ? <ArrowDown /> : <ArrowUpDown className="opacity-40" />}
    </Button>
  );
}

export default function RunsTable({
  source,
  notebook,
  rows,
  hyperparameterColumns,
}: {
  source: string;
  notebook: string;
  rows: RunTableRow[];
  hyperparameterColumns: { key: string; label: string }[];
}) {
  const router = useRouter();
  const [sorting, setSorting] = useState<SortingState>([]);
  const [selection, setSelection] = useState<RowSelectionState>({});

  const hyperparameterDefs: ColumnDef<RunTableRow>[] = hyperparameterColumns.map(({ key, label }) => ({
    id: `hp:${key}`,
    accessorFn: (row) => row.hyperparameters[key],
    header: ({ column }) => <SortHeader label={label} column={column} />,
    cell: ({ row }) => <HyperparameterValue value={row.original.hyperparameters[key]} />,
    sortUndefined: "last",
  }));

  const columns: ColumnDef<RunTableRow>[] = [
    {
      id: "select",
      header: ({ table }) => (
        <Checkbox
          aria-label="Select all runs"
          checked={table.getIsAllRowsSelected() ? true : table.getIsSomeRowsSelected() ? "indeterminate" : false}
          onCheckedChange={(value) => table.toggleAllRowsSelected(value === true)}
        />
      ),
      cell: ({ row }) => (
        <Checkbox aria-label={`Select ${row.original.name}`} checked={row.getIsSelected()} onCheckedChange={(value) => row.toggleSelected(value === true)} />
      ),
      enableSorting: false,
    },
    {
      accessorKey: "name",
      header: ({ column }) => <SortHeader label="run" column={column} />,
      cell: ({ row }) => (
        <Link href={runHref(source, notebook, row.original.name)} className="font-mono hover:underline">
          {row.original.name}
        </Link>
      ),
    },
    {
      accessorKey: "status",
      header: ({ column }) => <SortHeader label="status" column={column} />,
      cell: ({ row }) => <StatusBadge status={row.original.status} />,
    },
    ...hyperparameterDefs,
    {
      accessorKey: "stepsDone",
      header: ({ column }) => <SortHeader label="steps done / total" column={column} />,
      cell: ({ row }) => (
        <span className="tabular-nums">
          {row.original.stepsDone} / {row.original.totalSteps === null ? "?" : row.original.totalSteps}
        </span>
      ),
    },
    {
      accessorKey: "bestMetric",
      header: "best metric",
      cell: ({ row }) => (
        <span className="font-mono text-xs">
          {row.original.bestMetric} <span className="text-muted-foreground">({row.original.bestMode})</span>
        </span>
      ),
      enableSorting: false,
    },
    {
      accessorKey: "bestStep",
      header: ({ column }) => <SortHeader label="best step" column={column} />,
      cell: ({ row }) => <span className="tabular-nums">{row.original.bestStep === null ? "—" : row.original.bestStep}</span>,
      sortUndefined: "last",
    },
    {
      id: "bestValue",
      accessorFn: (row) => (row.bestValue === null ? undefined : row.bestValue),
      header: ({ column }) => <SortHeader label="best value" column={column} />,
      cell: ({ row }) => (
        <span className="tabular-nums">
          {formatNumber(row.original.bestValue)}
          {row.original.bestFromSummary || row.original.bestValue === null ? null : <span className="text-xs text-muted-foreground"> so far</span>}
        </span>
      ),
      sortUndefined: "last",
    },
    {
      accessorKey: "updated",
      header: ({ column }) => <SortHeader label="updated" column={column} />,
      cell: ({ row }) => (
        <span>
          <LocalTime ms={row.original.updated} /> <span className="text-xs text-muted-foreground">{row.original.age}</span>
        </span>
      ),
    },
  ];

  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, rowSelection: selection },
    getRowId: (row) => row.name,
    onSortingChange: setSorting,
    onRowSelectionChange: setSelection,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  // Compare keeps the notebook's run order so each run keeps its color slot; links carry the run source
  const selected = rows.filter((row) => selection[row.name] === true).map((row) => row.name);
  const tooMany = selected.length > MAX_SERIES;

  function openCompare() {
    const params = new URLSearchParams();
    for (const name of selected) {
      params.append("run", name);
    }
    router.push(compareHref(source, notebook, params.toString()));
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <Button size="sm" disabled={selected.length === 0 || tooMany} onClick={openCompare}>
          <GitCompare /> Compare {selected.length === 0 ? "selected runs" : `${selected.length} run${selected.length === 1 ? "" : "s"}`}
        </Button>
        <span className="text-xs text-muted-foreground">
          {tooMany
            ? `At most ${MAX_SERIES} runs can be compared in one chart.`
            : "Tick runs to compare. Total steps are known once a run has finished (run.json summary)."}
        </span>
      </div>
      <div className="overflow-hidden rounded-xl ring-1 ring-foreground/10">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id} className="first:pl-4">
                    {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id} data-state={row.getIsSelected() ? "selected" : undefined}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} className="first:pl-4">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
