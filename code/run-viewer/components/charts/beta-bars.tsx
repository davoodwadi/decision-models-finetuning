"use client";

import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { INK } from "@/components/charts/line-chart";
import { SERIES_DARK, SERIES_LIGHT, formatNumber, formatTick } from "@/lib/format";

export type BetaPair = { term: string; human: number; predicted: number };

type TooltipParam = { seriesName: string; color: string; value: number; axisValue: string };

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Horizontal grouped bars: per OLS term, the human-population beta (gray) next to the model-predicted beta (blue)
export default function BetaBars({ pairs }: { pairs: BetaPair[] }) {
  const { resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const dark = resolvedTheme === "dark";
  const ink = dark ? INK.dark : INK.light;
  const modelColor = (dark ? SERIES_DARK : SERIES_LIGHT)[0];
  const humanColor = dark ? "#8f8e86" : "#9a9993";
  // Category axes draw bottom-up; reverse so the first term is on top
  const ordered = [...pairs].reverse();
  const height = Math.max(160, ordered.length * 38 + 70);

  const option: echarts.EChartsOption = {
    backgroundColor: "transparent",
    animation: false,
    textStyle: { color: ink.text, fontFamily: "inherit" },
    grid: { left: 8, right: 24, top: 8, bottom: 40, containLabel: true },
    legend: {
      data: ["human population", "model"],
      bottom: 0,
      icon: "roundRect",
      itemWidth: 12,
      itemHeight: 8,
      textStyle: { color: ink.text, fontSize: 12 },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      backgroundColor: ink.tooltipBg,
      borderColor: ink.grid,
      textStyle: { color: ink.text, fontSize: 12 },
      formatter: (raw: unknown) => {
        const params = raw as TooltipParam[];
        if (params.length === 0) {
          return "";
        }
        const lines = [`<div style="margin-bottom:4px;max-width:360px;white-space:normal"><b>${escapeHtml(params[0].axisValue)}</b></div>`];
        for (const p of params) {
          lines.push(
            `<div style="display:flex;gap:8px;align-items:center"><span style="width:10px;height:8px;border-radius:2px;background:${p.color}"></span>` +
              `<span style="flex:1">${escapeHtml(p.seriesName)}</span><b style="font-variant-numeric:tabular-nums">${formatNumber(p.value)}</b></div>`,
          );
        }
        return lines.join("");
      },
    },
    xAxis: {
      type: "value",
      axisLabel: { color: ink.muted, fontSize: 11, formatter: (value: number) => formatTick(value) },
      splitLine: { lineStyle: { color: ink.grid } },
      axisLine: { show: true, lineStyle: { color: ink.axis } },
    },
    yAxis: {
      type: "category",
      data: ordered.map((pair) => pair.term),
      axisLabel: { color: ink.text, fontSize: 11, width: 300, overflow: "truncate" },
      axisLine: { lineStyle: { color: ink.axis } },
      axisTick: { show: false },
    },
    series: [
      {
        name: "human population",
        type: "bar",
        data: ordered.map((pair) => pair.human),
        itemStyle: { color: humanColor, borderRadius: 3 },
        barGap: "15%",
        barMaxWidth: 14,
      },
      {
        name: "model",
        type: "bar",
        data: ordered.map((pair) => pair.predicted),
        itemStyle: { color: modelColor, borderRadius: 3 },
        barMaxWidth: 14,
      },
    ],
  };

  if (!mounted) {
    return <div style={{ height }} />;
  }
  return <ReactECharts echarts={echarts} option={option} notMerge style={{ height, width: "100%" }} theme={dark ? "dark" : undefined} />;
}
