import { type NextRequest, NextResponse } from "next/server";
import { getSourceCheck } from "@/lib/sourceCheck";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const ds = request.nextUrl.searchParams.get("ds");
    if (!ds) {
      return NextResponse.json({ error: "ds is required" }, { status: 400 });
    }
    return NextResponse.json(await getSourceCheck(ds));
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
