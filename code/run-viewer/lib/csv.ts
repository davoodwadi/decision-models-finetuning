// CSV reading for files a training run may still be appending to.
// Rules:
// - an empty file or a header-only file gives zero rows (the run has not written any yet)
// - a last line without a trailing newline is a partial write: it is dropped
// - any other row whose field count differs from the header is an error (thrown, shown on the page)

export type Cell = number | string | null;

export type Table = {
  columns: string[];
  rows: Cell[][];
  droppedPartialLine: boolean;
};

function splitLine(line: string, path: string, lineNumber: number): string[] {
  const fields: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ",") {
      fields.push(current);
      current = "";
    } else {
      current += ch;
    }
  }
  if (inQuotes) {
    throw new Error(`${path}: line ${lineNumber} has an unclosed quote`);
  }
  fields.push(current);
  return fields;
}

function parseCell(text: string): Cell {
  // Empty and "nan" are missing values (pandas writes NaN as an empty field)
  if (text === "" || text === "nan" || text === "NaN") {
    return null;
  }
  const value = Number(text);
  if (Number.isNaN(value)) {
    return text;
  }
  return value;
}

export function parseCsv(text: string, path: string): Table {
  if (text.length === 0) {
    return { columns: [], rows: [], droppedPartialLine: false };
  }
  const lines = text.split("\n");
  // After a complete file, split leaves one empty string after the final newline
  const endsWithNewline = text.endsWith("\n");
  if (endsWithNewline) {
    lines.pop();
  }
  let droppedPartialLine = false;
  if (!endsWithNewline) {
    if (lines.length === 1) {
      // Only a partially written header: treat as an empty file
      return { columns: [], rows: [], droppedPartialLine: true };
    }
    lines.pop();
    droppedPartialLine = true;
  }
  const headerLine = lines[0].replace(/\r$/, "");
  const columns = splitLine(headerLine, path, 1);
  const rows: Cell[][] = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].replace(/\r$/, "");
    const fields = splitLine(line, path, i + 1);
    if (fields.length !== columns.length) {
      throw new Error(`${path}: line ${i + 1} has ${fields.length} fields, header has ${columns.length}`);
    }
    rows.push(fields.map(parseCell));
  }
  return { columns, rows, droppedPartialLine };
}

export function columnIndex(table: Table, column: string, path: string): number {
  const index = table.columns.indexOf(column);
  if (index === -1) {
    throw new Error(`${path}: missing column "${column}" (columns: ${table.columns.join(", ")})`);
  }
  return index;
}

export function numberAt(row: Cell[], index: number, column: string, path: string): number | null {
  const value = row[index];
  if (value !== null && typeof value !== "number") {
    throw new Error(`${path}: column "${column}" holds non-numeric value "${value}"`);
  }
  return value;
}
