"use client";

// Any error thrown while reading run files is shown here instead of being hidden
export default function ErrorPage({ error }: { error: Error & { digest?: string } }) {
  return (
    <div>
      <h1>Could not read the run files</h1>
      <div className="error">{error.message}</div>
    </div>
  );
}
