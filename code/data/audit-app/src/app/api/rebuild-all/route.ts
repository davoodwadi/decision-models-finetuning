import { NextResponse } from "next/server";
import { rebuildAllParquets } from "@/lib/store";

export async function POST() {
  try {
    const result = await rebuildAllParquets();
    return NextResponse.json(result, { status: result.ok ? 200 : 500 });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
