"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { useState } from "react";
import BetaBars from "@/components/charts/beta-bars";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { formatNumber } from "@/lib/format";

export type BetaRowView = { validator: string; model: string; term: string; human: number; predicted: number };
export type BetaFileView = { file: string; step: number; isBest: boolean; rows: BetaRowView[] };

function fileLabel(file: BetaFileView): string {
  return file.isBest ? `best checkpoint (step ${file.step})` : `step ${file.step}`;
}

function uniqueInOrder(values: string[]): string[] {
  const result: string[] = [];
  for (const value of values) {
    if (!result.includes(value)) {
      result.push(value);
    }
  }
  return result;
}

// OLS betas of the human population next to the OLS betas of the model's predicted choices, per validator and sub-model
export default function BetasView({ files }: { files: BetaFileView[] }) {
  // Best checkpoint first, then steps in order
  const ordered = [...files.filter((file) => file.isBest), ...files.filter((file) => !file.isBest).sort((a, b) => a.step - b.step)];
  const stepFiles = ordered.filter((file) => !file.isBest);
  const defaultFile = ordered[0].isBest ? ordered[0] : stepFiles[stepFiles.length - 1];
  const [fileName, setFileName] = useState(defaultFile.file);
  const allValidators = uniqueInOrder(ordered.flatMap((file) => file.rows.map((row) => row.validator)));
  const [validator, setValidator] = useState(allValidators[0]);

  const current = ordered.find((file) => file.file === fileName);
  if (current === undefined) {
    throw new Error(`No betas file ${fileName}`);
  }
  const position = stepFiles.findIndex((file) => file.file === fileName);
  const validatorsHere = uniqueInOrder(current.rows.map((row) => row.validator));
  const rows = current.rows.filter((row) => row.validator === validator);
  const models = uniqueInOrder(rows.map((row) => row.model));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">checkpoint</Label>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Previous step"
            disabled={position <= 0}
            onClick={() => setFileName(stepFiles[position - 1].file)}
          >
            <ChevronLeft />
          </Button>
          <Select value={fileName} onValueChange={setFileName}>
            <SelectTrigger size="sm" className="min-w-52 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ordered.map((file) => (
                <SelectItem key={file.file} value={file.file} className="text-xs">
                  {fileLabel(file)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Next step"
            disabled={position === -1 ? stepFiles.length === 0 : position >= stepFiles.length - 1}
            onClick={() => setFileName(stepFiles[position === -1 ? 0 : position + 1].file)}
          >
            <ChevronRight />
          </Button>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-xs text-muted-foreground">validator</Label>
          <ToggleGroup
            type="single"
            variant="outline"
            size="sm"
            value={validator}
            onValueChange={(value) => {
              if (value !== "") {
                setValidator(value);
              }
            }}
          >
            {allValidators.map((name) => (
              <ToggleGroupItem key={name} value={name} className="px-2.5 font-mono text-xs">
                {name}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </div>
        <span className="text-xs text-muted-foreground">betas/{current.file}</span>
      </div>

      {!validatorsHere.includes(validator) ? (
        <p className="text-sm text-muted-foreground">
          {validator} was not evaluated at {fileLabel(current)}; it ran at: {validatorsHere.join(", ")}.
        </p>
      ) : (
        models.map((model) => {
          const pairs = rows.filter((row) => row.model === model);
          return (
            <Card key={model} size="sm">
              <CardHeader>
                <CardTitle className="font-mono text-sm">{model}</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-1 gap-4 xl:grid-cols-[3fr_2fr]">
                <BetaBars pairs={pairs} />
                <div className="overflow-hidden rounded-lg ring-1 ring-foreground/10">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="pl-3">term</TableHead>
                        <TableHead className="text-right">human population β</TableHead>
                        <TableHead className="pr-3 text-right">model β</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pairs.map((pair) => (
                        <TableRow key={pair.term}>
                          <TableCell className="pl-3 text-xs whitespace-normal">{pair.term}</TableCell>
                          <TableCell className="text-right text-xs tabular-nums">{formatNumber(pair.human)}</TableCell>
                          <TableCell className="pr-3 text-right text-xs tabular-nums">{formatNumber(pair.predicted)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          );
        })
      )}
    </div>
  );
}
