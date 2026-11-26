import fs from "node:fs/promises";
import { type NextRequest, NextResponse } from "next/server";
import { getHighlightedPdf, getSourceDocumentPath } from "@/lib/sourceCheck";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  try {
    const sp = request.nextUrl.searchParams;
    const ds = sp.get("ds");
    const doc = sp.get("doc");
    if (!ds || !doc) {
      return NextResponse.json({ error: "ds and doc are required" }, { status: 400 });
    }
    // With `row`, serve the codebook with that question's block highlighted; without it, the file as is.
    const row = sp.get("row");
    const hasRev = Boolean(sp.get("rev"));
    const bytes = row ? await getHighlightedPdf(ds, doc, row) : await fs.readFile(await getSourceDocumentPath(ds, doc));
    return new NextResponse(new Uint8Array(bytes), {
      headers: {
        "Content-Type": "application/pdf",
        "Content-Disposition": "inline",
        "Cache-Control": hasRev ? "private, max-age=300" : "no-store",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
