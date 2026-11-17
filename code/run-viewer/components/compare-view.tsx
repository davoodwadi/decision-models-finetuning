"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import LineChart, { ChartSeries, XY } from "@/components/charts/line-chart";
import StatusBadge from "@/components/status-badge";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { SERIES_LIGHT, formatNumber, runHref } from "@/lib/format";

export type CompareRun = {
  name: string;
  // Palette slot = the run's position in the notebook, so a run keeps its color whatever else is selected
  slot: number;
  status: string;
  bestStep: number | null;
  metrics: Record<string, XY[]>;
};

export default function CompareView({
  notebook,
  runs,
  metricOptions,
  initialMetric,
}: {
  notebook: string;
  runs: CompareRun[];
  metricOptions: string[];
  initialMetric: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [metric, setMetric] = useState(initialMetric);

  function chooseMetric(value: string) {
    setMetric(value);
    // Keep the choice in the URL so a refresh or a shared link shows the same metric
    const params = new URLSearchParams(searchParams.toString());
    params.set("metric", value);
    router.replace(`${pathname}?${params.toString()}`, { scroll: false });
  }

  const series: ChartSeries[] = runs.map((run) => ({ name: run.name, slot: run.slot, points: run.metrics[metric] ?? [] }));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <Label className="text-xs text-muted-foreground">metric</Label>
        <Select value={metric} onValueChange={chooseMetric}>
          <SelectTrigger size="sm" className="min-w-64 font-mono text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {metricOptions.map((option) => (
              <SelectItem key={option} value={option} className="font-mono text-xs">
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <LineChart title={metric} series={series} xName="step" height={460} slider />
      <div className="overflow-hidden rounded-xl ring-1 ring-foreground/10">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="pl-4">run</TableHead>
              <TableHead>status</TableHead>
              <TableHead className="text-right">points</TableHead>
              <TableHead className="text-right">min</TableHead>
              <TableHead className="text-right">max</TableHead>
              <TableHead className="text-right">last</TableHead>
              <TableHead className="text-right">run&apos;s best step</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {runs.map((run) => {
              const points = run.metrics[metric];
              const values = points === undefined ? [] : points.map((point) => point[1]);
              return (
                <TableRow key={run.name}>
                  <TableCell className="pl-4">
                    <span className="inline-flex items-center gap-2">
                      <span className="h-1 w-3.5 rounded-full" style={{ background: SERIES_LIGHT[run.slot] }} />
                      <Link href={runHref(notebook, run.name)} className="font-mono hover:underline">
                        {run.name}
                      </Link>
                    </span>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={run.status} />
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {points === undefined ? <span className="text-xs text-muted-foreground">column absent</span> : values.length}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{values.length === 0 ? "" : formatNumber(Math.min(...values))}</TableCell>
                  <TableCell className="text-right tabular-nums">{values.length === 0 ? "" : formatNumber(Math.max(...values))}</TableCell>
                  <TableCell className="text-right tabular-nums">{values.length === 0 ? "" : formatNumber(values[values.length - 1])}</TableCell>
                  <TableCell className="text-right tabular-nums">{run.bestStep === null ? "—" : run.bestStep}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
