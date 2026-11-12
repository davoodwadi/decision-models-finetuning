import { type NextRequest, NextResponse } from "next/server";
import { type RowUpdateEntry, updateRowEntries, updateRows } from "@/lib/store";

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as {
      ds: string;
      rowIndices?: number[];
      patch?: Record<string, string>;
      reviewed?: boolean;
      entries?: RowUpdateEntry[];
    };
    if (!body.ds) {
      return NextResponse.json({ error: "ds is required" }, { status: 400 });
    }
    if (Array.isArray(body.entries) && body.entries.length > 0) {
      const result = await updateRowEntries(body.ds, body.entries);
      return NextResponse.json(result);
    }
    if (!Array.isArray(body.rowIndices) || body.rowIndices.length === 0) {
      return NextResponse.json(
        { error: "non-empty rowIndices or entries are required" },
        { status: 400 }
      );
    }
    const result = await updateRows(
      body.ds,
      body.rowIndices,
      body.patch ?? {},
      body.reviewed
    );
    return NextResponse.json(result);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 400 });
  }
}
