import { NextResponse } from "next/server";
import { discoverDatasets } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const result = await discoverDatasets();
    return NextResponse.json(result);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
