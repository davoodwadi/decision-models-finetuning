"use client";

import { useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { SERIES_COLORS, formatNumber } from "@/lib/format";

export type Point = { x: number; y: number };
export type Series = { name: string; points: Point[] };

// One line per series on a shared numeric x axis (steps). Series may have points at different x.
export default function MetricChart({
  title,
  series,
  xLabel,
  markX,
  height = 260,
}: {
  title: string;
  series: Series[];
  xLabel: string;
  markX?: number | null;
  height?: number;
}) {
  const [logScale, setLogScale] = useState(false);
  let allPositive = true;
  let pointCount = 0;
  for (const s of series) {
    for (const p of s.points) {
      pointCount++;
      if (p.y <= 0) {
        allPositive = false;
      }
    }
  }
  const useLog = logScale && allPositive;

  return (
    <div className="card chart">
      <div className="chart-head">
        <h3>{title}</h3>
        <label className="muted small">
          <input
            type="checkbox"
            checked={useLog}
            disabled={!allPositive}
            onChange={(event) => setLogScale(event.target.checked)}
          />{" "}
          log y{allPositive ? "" : " (values ≤ 0)"}
        </label>
      </div>
      {pointCount === 0 ? (
        <p className="muted">No values yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={height}>
          <LineChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="var(--grid)" strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="x"
              domain={["dataMin", "dataMax"]}
              allowDuplicatedCategory={false}
              tick={{ fill: "var(--muted)", fontSize: 12 }}
              label={{ value: xLabel, position: "insideBottomRight", offset: -4, fill: "var(--muted)", fontSize: 12 }}
            />
            <YAxis
              type="number"
              scale={useLog ? "log" : "auto"}
              domain={["auto", "auto"]}
              tickFormatter={(value: number) => formatNumber(value)}
              tick={{ fill: "var(--muted)", fontSize: 12 }}
              width={72}
            />
            <Tooltip
              formatter={(value) => formatNumber(typeof value === "number" ? value : null)}
              labelFormatter={(label) => `${xLabel} ${label}`}
              contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", fontSize: 12 }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {series.map((s, i) => (
              <Line
                key={s.name}
                name={s.name}
                data={s.points}
                dataKey="y"
                stroke={SERIES_COLORS[i % SERIES_COLORS.length]}
                dot={s.points.length < 40 ? { r: 2 } : false}
                strokeWidth={1.6}
                isAnimationActive={false}
              />
            ))}
            {markX !== undefined && markX !== null ? (
              <ReferenceLine
                x={markX}
                stroke="var(--best)"
                strokeDasharray="4 3"
                label={{ value: `best ${markX}`, position: "insideTopRight", fill: "var(--muted)", fontSize: 11 }}
              />
            ) : null}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
