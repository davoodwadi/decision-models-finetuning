import { CircleCheck, CirclePause, LoaderCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

// Status always carries an icon and a label, never color alone
export default function StatusBadge({ status }: { status: string }) {
  if (status === "running") {
    return (
      <Badge variant="outline" className="border-emerald-600/40 text-emerald-700 dark:text-emerald-400">
        <LoaderCircle className="animate-spin" /> running
      </Badge>
    );
  }
  if (status === "interrupted") {
    return (
      <Badge variant="outline" className="border-amber-600/40 text-amber-700 dark:text-amber-400">
        <CirclePause /> interrupted
      </Badge>
    );
  }
  if (status === "finished") {
    return (
      <Badge variant="secondary">
        <CircleCheck /> finished
      </Badge>
    );
  }
  throw new Error(`Unknown run status "${status}"`);
}
