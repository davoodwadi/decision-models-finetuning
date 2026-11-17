"use client";

import { useState } from "react";
import MetricChart, { Point, Series } from "./MetricChart";

export type StepGroup = {
  name: string;
  steps: number[];
  values: Record<string, (number | null)[]>;
};

const WINDOWS = [1, 5, 10, 25, 50, 100];

// Trailing mean over the last `window` rows of one group; missing values inside the window are skipped
function rollingMean(steps: number[], values: (number | null)[], window: number): Point[] {
  const points: Point[] = [];
  for (let i = 0; i < steps.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - window + 1); j <= i; j++) {
      const value = values[j];
      if (value !== null) {
        sum += value;
        count++;
      }
    }
    if (count > 0 && values[i] !== null) {
      points.push({ x: steps[i], y: sum / count });
    }
  }
  return points;
}

// Training-step chart: one line per group (dataset), column and rolling window chosen by the viewer
export default function RollingChart({
  title,
  columns,
  groups,
  defaultWindow,
}: {
  title: string;
  columns: string[];
  groups: StepGroup[];
  defaultWindow: number;
}) {
  const [column, setColumn] = useState(columns[0]);
  const [window, setWindow] = useState(defaultWindow);
  const series: Series[] = [];
  for (const group of groups) {
    series.push({ name: group.name, points: rollingMean(group.steps, group.values[column], window) });
  }
  return (
    <div>
      <div className="controls">
        <label>
          column{" "}
          <select value={column} onChange={(event) => setColumn(event.target.value)}>
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label>
          rolling mean over{" "}
          <select value={window} onChange={(event) => setWindow(Number(event.target.value))}>
            {WINDOWS.map((w) => (
              <option key={w} value={w}>
                {w === 1 ? "1 (raw)" : w}
              </option>
            ))}
          </select>{" "}
          steps of each dataset
        </label>
      </div>
      <MetricChart title={`${title}: ${column}`} series={series} xLabel="step" height={320} />
    </div>
  );
}
