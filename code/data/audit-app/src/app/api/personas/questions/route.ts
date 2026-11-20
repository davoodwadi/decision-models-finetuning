import { type NextRequest, NextResponse } from "next/server";
import { discoverDatasets, getRespondentQuestions } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const sp = request.nextUrl.searchParams;
    let ds = sp.get("ds");
    if (!ds) {
      const { defaultDs } = await discoverDatasets();
      ds = defaultDs;
    }
    const caseId = sp.get("caseId");
    if (!caseId) {
      return NextResponse.json(
        { error: "Missing required query parameter 'caseId'" },
        { status: 400 }
      );
    }
    const limit = Number(sp.get("limit") ?? "50") || 50;
    const payload = await getRespondentQuestions(ds, caseId, limit);
    return NextResponse.json(payload);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
