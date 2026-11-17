"use client";

import { CircleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

// Any error thrown while reading run files is shown here instead of being hidden
export default function ErrorPage({ error }: { error: Error & { digest?: string } }) {
  return (
    <Alert variant="destructive">
      <CircleAlert />
      <AlertTitle>Could not read the run files</AlertTitle>
      <AlertDescription className="font-mono text-xs whitespace-pre-wrap">{error.message}</AlertDescription>
    </Alert>
  );
}
