import { formatTick } from "@/lib/format";

// One main hyperparameter in a table cell; undefined = the run's config has no such key
export default function HyperparameterValue({ value }: { value: string | number | undefined }) {
  if (value === undefined) {
    return <span className="text-muted-foreground">—</span>;
  }
  if (typeof value === "number") {
    return <span className="tabular-nums">{formatTick(value)}</span>;
  }
  return <span className="font-mono text-xs">{value}</span>;
}
