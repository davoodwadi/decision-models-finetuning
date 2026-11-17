"use client";

import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { INK } from "@/lib/chart-theme";
import { MAX_SERIES, SERIES_DARK, SERIES_LIGHT, formatNumber, formatTick } from "@/lib/format";

export type XY = [number, number];

export type ChartSeries = {
  name: string;
  // Fixed palette slot for this entity (never its rank in the current selection)
  slot: number;
  points: XY[];
  // Optional unsmoothed values drawn faintly behind `points` (same legend entry)
  raw?: XY[];
};


function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

type TooltipParam = { seriesId: string; seriesName: string; color: string; value: XY };

// Line chart for training curves: crosshair tooltip, ctrl+wheel / drag zoom, box zoom and reset in the toolbar,
// click a legend entry to hide a series, optional log y. Charts sharing `group` move their crosshair and zoom together.
export default function LineChart({
  title,
  description,
  series,
  xName,
  markX,
  group,
  height = 280,
  slider = false,
}: {
  title: string;
  description?: string;
  series: ChartSeries[];
  xName: string;
  markX?: number | null;
  group?: string;
  height?: number;
  slider?: boolean;
}) {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [logY, setLogY] = useState(false);
  useEffect(() => setMounted(true), []);

  for (const s of series) {
    if (s.slot < 0 || s.slot >= MAX_SERIES) {
      throw new Error(`Chart "${title}": series "${s.name}" has palette slot ${s.slot}; at most ${MAX_SERIES} series can be told apart`);
    }
  }

  let pointCount = 0;
  let allPositive = true;
  for (const s of series) {
    for (const [, y] of [...s.points, ...(s.raw ?? [])]) {
      pointCount++;
      if (y <= 0) {
        allPositive = false;
      }
    }
  }
  const useLog = logY && allPositive;
  const dark = resolvedTheme === "dark";
  const ink = dark ? INK.dark : INK.light;
  const palette = dark ? SERIES_DARK : SERIES_LIGHT;

  const echartsSeries: echarts.SeriesOption[] = [];
  for (const s of series) {
    const color = palette[s.slot];
    if (s.raw !== undefined) {
      echartsSeries.push({
        id: `${s.name}::raw`,
        name: s.name,
        type: "line",
        data: s.raw,
        showSymbol: false,
        lineStyle: { width: 1, opacity: 0.22, color },
        itemStyle: { color },
        emphasis: { disabled: true },
        silent: true,
        z: 1,
      });
    }
    echartsSeries.push({
      id: s.name,
      name: s.name,
      type: "line",
      data: s.points,
      showSymbol: s.points.length <= 40,
      symbolSize: 7,
      lineStyle: { width: 2, color },
      itemStyle: { color, borderColor: dark ? "#1a1a19" : "#fcfcfb", borderWidth: 2 },
      emphasis: { focus: "series" },
      z: 2,
    });
  }
  if (markX !== undefined && markX !== null && echartsSeries.length > 0) {
    echartsSeries.push({
      id: "::best",
      name: "best step",
      type: "line",
      data: [],
      silent: true,
      markLine: {
        symbol: "none",
        silent: true,
        lineStyle: { type: "dashed", color: ink.best, width: 1, opacity: 0.6 },
        label: { formatter: `best ${markX}`, color: ink.muted, fontSize: 11, position: "insideEndTop" },
        data: [{ xAxis: markX }],
      },
    });
  }

  const legendNames = series.map((s) => s.name);
  const option: echarts.EChartsOption = {
    backgroundColor: "transparent",
    animation: false,
    textStyle: { color: ink.text, fontFamily: "inherit" },
    grid: { left: 8, right: 16, top: 36, bottom: slider ? 64 : 36, containLabel: true },
    legend: {
      data: legendNames,
      type: "scroll",
      bottom: slider ? 30 : 0,
      icon: "roundRect",
      itemWidth: 14,
      itemHeight: 4,
      textStyle: { color: ink.text, fontSize: 12 },
      pageTextStyle: { color: ink.text },
    },
    toolbox: {
      right: 4,
      top: 0,
      itemSize: 13,
      iconStyle: { borderColor: ink.muted },
      feature: {
        dataZoom: { yAxisIndex: "none", title: { zoom: "box zoom", back: "undo zoom" } },
        restore: { title: "reset" },
        saveAsImage: { title: "save PNG", backgroundColor: dark ? "#1a1a19" : "#ffffff" },
      },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: ink.axis } },
      backgroundColor: ink.tooltipBg,
      borderColor: ink.grid,
      textStyle: { color: ink.text, fontSize: 12 },
      formatter: (raw: unknown) => {
        const params = raw as TooltipParam[];
        if (params.length === 0) {
          return "";
        }
        const rawValues = new Map<string, number>();
        for (const p of params) {
          if (p.seriesId.endsWith("::raw")) {
            rawValues.set(p.seriesName, p.value[1]);
          }
        }
        const lines = [`<div style="margin-bottom:4px">${escapeHtml(xName)} <b>${params[0].value[0]}</b></div>`];
        for (const p of params) {
          if (p.seriesId.endsWith("::raw") || p.seriesId === "::best") {
            continue;
          }
          const rawValue = rawValues.get(p.seriesName);
          const rawText = rawValue === undefined ? "" : ` <span style="opacity:.6">(raw ${formatNumber(rawValue)})</span>`;
          lines.push(
            `<div style="display:flex;gap:8px;align-items:center"><span style="width:10px;height:3px;border-radius:2px;background:${p.color}"></span>` +
              `<span style="flex:1">${escapeHtml(p.seriesName)}</span><b style="font-variant-numeric:tabular-nums">${formatNumber(p.value[1])}</b>${rawText}</div>`,
          );
        }
        return lines.join("");
      },
    },
    xAxis: {
      type: "value",
      name: xName,
      nameLocation: "end",
      nameTextStyle: { color: ink.muted, fontSize: 11 },
      min: "dataMin",
      max: "dataMax",
      axisLine: { lineStyle: { color: ink.axis } },
      axisLabel: { color: ink.muted, fontSize: 11 },
      splitLine: { show: false },
    },
    yAxis: {
      type: useLog ? "log" : "value",
      scale: true,
      axisLabel: { color: ink.muted, fontSize: 11, formatter: (value: number) => formatTick(value) },
      splitLine: { lineStyle: { color: ink.grid } },
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, filterMode: "none", zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false, moveOnMouseMove: true },
      ...(slider
        ? [{ type: "slider" as const, xAxisIndex: 0, filterMode: "none" as const, height: 18, bottom: 4, borderColor: ink.grid, textStyle: { color: ink.muted } }]
        : []),
    ],
    series: echartsSeries,
  };

  return (
    <Card size="sm" className="gap-2">
      <CardHeader>
        <CardTitle className="font-mono text-sm">{title}</CardTitle>
        {description === undefined ? null : <CardDescription>{description}</CardDescription>}
        <CardAction>
          <div className="flex items-center gap-2">
            <Switch id={`log-${title}`} size="sm" checked={useLog} disabled={!allPositive} onCheckedChange={setLogY} />
            <Label htmlFor={`log-${title}`} className="text-xs text-muted-foreground">
              log y
            </Label>
          </div>
        </CardAction>
      </CardHeader>
      <CardContent>
        {pointCount === 0 ? (
          <p className="text-sm text-muted-foreground">No values yet.</p>
        ) : !mounted ? (
          <div style={{ height }} />
        ) : (
          <ReactECharts
            echarts={echarts}
            option={option}
            notMerge
            style={{ height, width: "100%" }}
            theme={dark ? "dark" : undefined}
            onChartReady={(chart: echarts.ECharts) => {
              if (group !== undefined) {
                chart.group = group;
                echarts.connect(group);
              }
            }}
          />
        )}
      </CardContent>
    </Card>
  );
}
