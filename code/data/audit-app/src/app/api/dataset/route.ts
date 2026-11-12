import { type NextRequest, NextResponse } from "next/server";
import { discoverDatasets, getDatasetPayload } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    let ds = searchParams.get("ds");
    if (!ds) {
      const { defaultDs } = await discoverDatasets();
      ds = defaultDs;
    }
    const payload = await getDatasetPayload(ds);
    return NextResponse.json(payload);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
