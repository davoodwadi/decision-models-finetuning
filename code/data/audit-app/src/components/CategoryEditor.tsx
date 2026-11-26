"use client";

import * as React from "react";
import { Check, Folder, Pencil, Plus, XIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

interface CategoryEditorProps {
  category: string;
  allCategoryCounts: Array<[string, number]>;
  onChange: (nextCategory: string) => void;
  onFilterByCategory?: (category: string) => void;
}

export function CategoryEditor({
  category,
  allCategoryCounts,
  onChange,
  onFilterByCategory,
}: CategoryEditorProps) {
  const [open, setOpen] = React.useState(false);
  const [draftInput, setDraftInput] = React.useState("");
  const inputRef = React.useRef<HTMLInputElement>(null);

  const trimmedCurrent = category.trim();
  const trimmedInput = draftInput.trim();

  const filteredKnownCategories = React.useMemo(() => {
    const q = trimmedInput.toLowerCase();
    if (!q) return allCategoryCounts;
    return allCategoryCounts.filter(([cat]) => cat.toLowerCase().includes(q));
  }, [allCategoryCounts, trimmedInput]);

  const exactKnownMatch = React.useMemo(() => {
    if (!trimmedInput) return null;
    const lower = trimmedInput.toLowerCase();
    return allCategoryCounts.find(([cat]) => cat.toLowerCase() === lower)?.[0] ?? null;
  }, [allCategoryCounts, trimmedInput]);

  const selectCategory = (next: string) => {
    onChange(next.trim());
    setDraftInput("");
    setOpen(false);
  };

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <span className="inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-0.5">
        <Folder className="w-3 h-3" />
        Category:
      </span>

      {trimmedCurrent && (
        <Badge
          variant="outline"
          className="h-5.5 pl-2 pr-1 py-0 text-[11px] font-medium bg-teal-50 text-teal-800 border-teal-200 rounded-md inline-flex items-center gap-1"
        >
          {onFilterByCategory ? (
            <button
              type="button"
              onClick={() => onFilterByCategory(trimmedCurrent)}
              title={`Filter sidebar by category "${trimmedCurrent}"`}
              className="hover:underline cursor-pointer"
            >
              {trimmedCurrent}
            </button>
          ) : (
            <span>{trimmedCurrent}</span>
          )}
          <button
            type="button"
            aria-label={`Clear category ${trimmedCurrent}`}
            onClick={() => onChange("")}
            title="Clear category"
            className="rounded p-0.5 text-teal-500 hover:bg-teal-200/70 hover:text-teal-900 cursor-pointer"
          >
            <XIcon className="w-2.5 h-2.5" />
          </button>
        </Badge>
      )}

      <Popover
        open={open}
        onOpenChange={(nextOpen) => {
          setOpen(nextOpen);
          if (nextOpen) {
            requestAnimationFrame(() => inputRef.current?.focus());
          } else {
            setDraftInput("");
          }
        }}
      >
        <PopoverTrigger
          className={cn(
            buttonVariants({ variant: "outline", size: "xs" }),
            "h-5.5 px-2 text-[11px] bg-white text-slate-600 border-dashed border-slate-300 hover:border-teal-300 hover:text-teal-800 hover:bg-teal-50/40 shadow-none cursor-pointer"
          )}
        >
          {trimmedCurrent ? (
            <>
              <Pencil className="w-2.5 h-2.5" />
              Change
            </>
          ) : (
            <>
              <Plus className="w-3 h-3" />
              Set category
            </>
          )}
        </PopoverTrigger>

        <PopoverContent align="start" className="w-72 p-2 space-y-2">
          <div className="space-y-1">
            <Input
              ref={inputRef}
              type="text"
              value={draftInput}
              onChange={(e) => setDraftInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.stopPropagation();
                  if (trimmedInput) {
                    selectCategory(exactKnownMatch ?? trimmedInput);
                  }
                } else if (e.key === "Escape") {
                  e.stopPropagation();
                  setOpen(false);
                }
              }}
              placeholder="Search or create category..."
              className="h-7 text-xs bg-white shadow-none"
            />
            {trimmedInput && !exactKnownMatch && (
              <button
                type="button"
                onClick={() => selectCategory(trimmedInput)}
                className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs font-medium bg-teal-50 text-teal-800 hover:bg-teal-100 border border-teal-200/80 text-left cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">
                  Create <strong>{trimmedInput}</strong>
                </span>
              </button>
            )}
          </div>

          {filteredKnownCategories.length > 0 ? (
            <div className="max-h-56 overflow-y-auto space-y-0.5 pt-1 border-t border-slate-100">
              <div className="px-1.5 pb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Categories in dataset
              </div>
              {filteredKnownCategories.map(([cat, count]) => {
                const selected =
                  trimmedCurrent.toLowerCase() === cat.toLowerCase();
                return (
                  <button
                    key={cat}
                    type="button"
                    onClick={() => selectCategory(cat)}
                    className={cn(
                      "w-full flex items-center gap-2 px-1.5 py-1 rounded-md text-xs text-left cursor-pointer",
                      selected
                        ? "bg-teal-50 text-teal-900 font-medium"
                        : "hover:bg-slate-100 text-slate-800"
                    )}
                  >
                    <Check
                      className={cn(
                        "w-3.5 h-3.5 shrink-0",
                        selected ? "text-teal-700 opacity-100" : "opacity-0"
                      )}
                    />
                    <span className="truncate flex-1">{cat}</span>
                    <span className="font-mono text-[11px] text-slate-400">
                      {count}
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            !trimmedInput && (
              <div className="py-2 text-center text-[11px] text-slate-400">
                No categories in this dataset yet — type a name above and press
                Enter.
              </div>
            )
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
