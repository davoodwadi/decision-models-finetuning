import { type NextRequest, NextResponse } from "next/server";
import { discoverDatasets, getQuestionMarginals } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const sp = request.nextUrl.searchParams;
    let ds = sp.get("ds");
    if (!ds) {
      const { defaultDs } = await discoverDatasets();
      ds = defaultDs;
    }
    const questionId = sp.get("questionId")?.trim();
    if (!questionId) {
      return NextResponse.json(
        { error: "Missing required query parameter 'questionId'" },
        { status: 400 }
      );
    }
    const subitemId = sp.get("subitemId")?.trim() || undefined;
    const study = sp.get("study")?.trim() || undefined;
    const persona = sp.get("persona")?.trim() || undefined;
    const arm = sp.get("arm")?.trim() || undefined;

    const payload = await getQuestionMarginals(ds, {
      questionId,
      subitemId,
      study,
      persona,
      arm,
    });
    return NextResponse.json(payload);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
