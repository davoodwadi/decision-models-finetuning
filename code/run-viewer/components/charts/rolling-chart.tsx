"use client";

import { useState } from "react";
import LineChart, { ChartSeries, XY } from "@/components/charts/line-chart";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";

export type StepGroup = {
  name: string;
  // Fixed palette slot of this dataset
  slot: number;
  steps: number[];
  values: Record<string, (number | null)[]>;
};

const WINDOWS = [1, 10, 25, 50, 100];

// Trailing mean over the last `window` rows of one dataset; missing values inside the window are skipped
function rollingMean(steps: number[], values: (number | null)[], window: number): XY[] {
  const points: XY[] = [];
  for (let i = 0; i < steps.length; i++) {
    if (values[i] === null) {
      continue;
    }
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) {
      const value = values[j];
      if (value !== null) {
        sum += value;
        count++;
      }
    }
    points.push([steps[i], sum / count]);
  }
  return points;
}

function rawPoints(steps: number[], values: (number | null)[]): XY[] {
  const points: XY[] = [];
  for (let i = 0; i < steps.length; i++) {
    const value = values[i];
    if (value !== null) {
      points.push([steps[i], value]);
    }
  }
  return points;
}

// Per-training-step chart: one line per dataset; the viewer picks the column, the rolling window and whether raw values show
export default function RollingChart({
  title,
  columns,
  groups,
  defaultWindow,
  height = 380,
}: {
  title: string;
  columns: string[];
  groups: StepGroup[];
  defaultWindow: number;
  height?: number;
}) {
  const [column, setColumn] = useState(columns[0]);
  const [window, setWindow] = useState(defaultWindow);
  const [showRaw, setShowRaw] = useState(true);

  const series: ChartSeries[] = [];
  for (const group of groups) {
    const values = group.values[column];
    series.push({
      name: group.name,
      slot: group.slot,
      points: rollingMean(group.steps, values, window),
      raw: showRaw && window > 1 ? rawPoints(group.steps, values) : undefined,
    });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">column</Label>
          <Select value={column} onValueChange={setColumn}>
            <SelectTrigger size="sm" className="font-mono text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {columns.map((c) => (
                <SelectItem key={c} value={c} className="font-mono text-xs">
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">rolling mean (steps per dataset)</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={String(window)}
            onValueChange={(value) => {
              if (value !== "") {
                setWindow(Number(value));
              }
            }}
          >
            {WINDOWS.map((w) => (
              <ToggleGroupItem key={w} value={String(w)} className="px-2.5 text-xs">
                {w === 1 ? "raw" : w}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        <div className="flex items-center gap-2">
          <Switch id={`raw-${title}`} size="sm" checked={showRaw} disabled={window === 1} onCheckedChange={setShowRaw} />
          <Label htmlFor={`raw-${title}`} className="text-xs text-muted-foreground">
            show raw behind mean
          </Label>
        </div>
      </div>
      <LineChart title={`${title}: ${column}`} series={series} xName="step" height={height} slider />
    </div>
  );
}
