import { type NextRequest, NextResponse } from "next/server";
import { discoverDatasets, getPersonasPage } from "@/lib/store";
import type { PersonaCompletenessFilter } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const sp = request.nextUrl.searchParams;
    let ds = sp.get("ds");
    if (!ds) {
      const { defaultDs } = await discoverDatasets();
      ds = defaultDs;
    }

    const search = sp.get("search") ?? "";
    const attrCol = sp.get("attrCol") ?? "";
    const attrVal = sp.get("attrVal") ?? "";
    const rawComp = sp.get("completeness") ?? "all";
    const completeness: PersonaCompletenessFilter =
      rawComp === "complete" || rawComp === "incomplete" ? rawComp : "all";
    const page = Number(sp.get("page") ?? "1") || 1;
    const pageSize = Number(sp.get("pageSize") ?? "100") || 100;

    const payload = await getPersonasPage(ds, {
      search,
      attrCol,
      attrVal,
      completeness,
      page,
      pageSize,
    });
    return NextResponse.json(payload);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
