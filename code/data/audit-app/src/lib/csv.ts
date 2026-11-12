/**
 * Strict RFC-4180 CSV parser and serializer matching pandas.read_csv(..., keep_default_na=False)
 * and DataFrame.to_csv(..., index=False) byte-for-byte.
 */

export function parseCsv(content: string): {
  columns: string[];
  rows: Record<string, string>[];
} {
  const records: string[][] = [];
  let curField = "";
  let curRow: string[] = [];
  let inQuotes = false;
  const n = content.length;

  for (let i = 0; i < n; i++) {
    const ch = content[i];
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < n && content[i + 1] === '"') {
          curField += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        curField += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        curRow.push(curField);
        curField = "";
      } else if (ch === "\r") {
        if (i + 1 < n && content[i + 1] === "\n") {
          i++;
        }
        curRow.push(curField);
        curField = "";
        records.push(curRow);
        curRow = [];
      } else if (ch === "\n") {
        curRow.push(curField);
        curField = "";
        records.push(curRow);
        curRow = [];
      } else {
        curField += ch;
      }
    }
  }

  if (inQuotes) {
    throw new Error("Malformed CSV: unterminated quoted field");
  }
  if (curField.length > 0 || curRow.length > 0) {
    curRow.push(curField);
    records.push(curRow);
  }
  if (records.length === 0) {
    throw new Error("Empty CSV file");
  }

  const columns = records[0];
  const rows: Record<string, string>[] = [];
  for (let r = 1; r < records.length; r++) {
    const vals = records[r];
    if (vals.length === 1 && vals[0] === "" && columns.length > 1) {
      continue;
    }
    if (vals.length !== columns.length) {
      throw new Error(
        `CSV row ${r} has ${vals.length} fields, expected ${columns.length}`
      );
    }
    const obj: Record<string, string> = {};
    for (let c = 0; c < columns.length; c++) {
      obj[columns[c]] = vals[c];
    }
    rows.push(obj);
  }

  return { columns, rows };
}

function escapeCsvField(val: string): string {
  if (
    val.includes(",") ||
    val.includes('"') ||
    val.includes("\n") ||
    val.includes("\r")
  ) {
    return `"${val.replaceAll('"', '""')}"`;
  }
  return val;
}

export function stringifyCsv(
  columns: string[],
  rows: Record<string, string>[]
): string {
  const lines: string[] = [columns.map(escapeCsvField).join(",")];
  for (const row of rows) {
    const encoded = columns.map((col) => {
      if (!(col in row)) {
        throw new Error(`Missing expected column ${col} during CSV serialization`);
      }
      return escapeCsvField(row[col]);
    });
    lines.push(encoded.join(","));
  }
  return lines.join("\n") + "\n";
}
