import { type NextRequest, NextResponse } from "next/server";
import { rebuildDatasetParquet } from "@/lib/store";

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as { ds: string };
    if (!body.ds) {
      return NextResponse.json({ error: "ds is required" }, { status: 400 });
    }
    const result = await rebuildDatasetParquet(body.ds);
    const status = result.ok ? 200 : result.blocked ? 409 : 500;
    return NextResponse.json(result, { status });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
