"use client";

import * as React from "react";
import { Plus, Tag, XIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { normalizeTag, normalizeTagList } from "@/lib/triage";
import { cn } from "@/lib/utils";

interface TagEditorProps {
  tags: string[];
  allTagCounts: Array<[string, number]>;
  onChange: (nextTags: string[]) => void;
  onFilterByTag?: (tag: string) => void;
}

export function TagEditor({
  tags,
  allTagCounts,
  onChange,
  onFilterByTag,
}: TagEditorProps) {
  const [open, setOpen] = React.useState(false);
  const [draftInput, setDraftInput] = React.useState("");
  const inputRef = React.useRef<HTMLInputElement>(null);

  const lowerToActive = React.useMemo(() => {
    const map = new Map<string, string>();
    for (const t of tags) {
      map.set(t.toLowerCase(), t);
    }
    return map;
  }, [tags]);

  const parsedTypedTags = React.useMemo(() => {
    if (!draftInput.trim()) return [];
    return normalizeTagList(draftInput.split(","));
  }, [draftInput]);

  const filteredKnownTags = React.useMemo(() => {
    const q = normalizeTag(draftInput).toLowerCase();
    if (!q) return allTagCounts;
    return allTagCounts.filter(([t]) => t.toLowerCase().includes(q));
  }, [allTagCounts, draftInput]);

  const newCreatableTags = React.useMemo(() => {
    const knownLower = new Set(allTagCounts.map(([t]) => t.toLowerCase()));
    return parsedTypedTags.filter(
      (t) =>
        !knownLower.has(t.toLowerCase()) && !lowerToActive.has(t.toLowerCase())
    );
  }, [parsedTypedTags, allTagCounts, lowerToActive]);

  const addTags = (incoming: string[]) => {
    if (incoming.length === 0) return;
    const knownCasing = new Map(
      allTagCounts.map(([t]) => [t.toLowerCase(), t] as const)
    );
    const mapped = incoming.map((t) => knownCasing.get(t.toLowerCase()) ?? t);
    onChange(normalizeTagList([...tags, ...mapped]));
    setDraftInput("");
  };

  const toggleTag = (tag: string) => {
    const lower = tag.toLowerCase();
    if (lowerToActive.has(lower)) {
      onChange(tags.filter((t) => t.toLowerCase() !== lower));
    } else {
      onChange(normalizeTagList([...tags, tag]));
    }
  };

  const removeTag = (tag: string) => {
    const lower = tag.toLowerCase();
    onChange(tags.filter((t) => t.toLowerCase() !== lower));
  };

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      <span className="inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider text-slate-400 mr-0.5">
        <Tag className="w-3 h-3" />
        Tags:
      </span>

      {tags.map((tag) => (
        <Badge
          key={tag}
          variant="outline"
          className="h-5.5 pl-2 pr-1 py-0 text-[11px] font-medium bg-indigo-50 text-indigo-800 border-indigo-200 rounded-md inline-flex items-center gap-1"
        >
          {onFilterByTag ? (
            <button
              type="button"
              onClick={() => onFilterByTag(tag)}
              title={`Filter sidebar by #${tag}`}
              className="hover:underline cursor-pointer font-mono"
            >
              #{tag}
            </button>
          ) : (
            <span className="font-mono">#{tag}</span>
          )}
          <button
            type="button"
            aria-label={`Remove tag ${tag}`}
            onClick={() => removeTag(tag)}
            title={`Remove #${tag}`}
            className="rounded p-0.5 text-indigo-500 hover:bg-indigo-200/70 hover:text-indigo-900 cursor-pointer"
          >
            <XIcon className="w-2.5 h-2.5" />
          </button>
        </Badge>
      ))}

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
            "h-5.5 px-2 text-[11px] bg-white text-slate-600 border-dashed border-slate-300 hover:border-indigo-300 hover:text-indigo-800 hover:bg-indigo-50/40 shadow-none cursor-pointer"
          )}
        >
          <Plus className="w-3 h-3" />
          Add tag
        </PopoverTrigger>

        <PopoverContent align="start" className="w-64 p-2 space-y-2">
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
                  if (parsedTypedTags.length > 0) {
                    addTags(parsedTypedTags);
                  }
                } else if (e.key === "Escape") {
                  e.stopPropagation();
                  setOpen(false);
                }
              }}
              placeholder="Type tag & press Enter..."
              className="h-7 text-xs bg-white shadow-none font-mono"
            />
            {newCreatableTags.length > 0 && (
              <button
                type="button"
                onClick={() => addTags(parsedTypedTags)}
                className="w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-xs font-medium bg-indigo-50 text-indigo-800 hover:bg-indigo-100 border border-indigo-200/80 text-left cursor-pointer"
              >
                <Plus className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">
                  Create{" "}
                  <strong className="font-mono">
                    {newCreatableTags.map((t) => `#${t}`).join(", ")}
                  </strong>
                </span>
              </button>
            )}
          </div>

          {filteredKnownTags.length > 0 ? (
            <div className="max-h-48 overflow-y-auto space-y-0.5 pt-1 border-t border-slate-100">
              <div className="px-1.5 pb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Existing tags in dataset
              </div>
              {filteredKnownTags.map(([tag, count]) => {
                const checked = lowerToActive.has(tag.toLowerCase());
                return (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => toggleTag(tag)}
                    className="w-full flex items-center gap-2 px-1.5 py-1 rounded-md hover:bg-slate-100 text-xs text-left cursor-pointer"
                  >
                    <Checkbox
                      checked={checked}
                      tabIndex={-1}
                      className="pointer-events-none"
                    />
                    <span className="font-mono text-slate-800 truncate flex-1">
                      #{tag}
                    </span>
                    <span className="font-mono text-[11px] text-slate-400">
                      {count}
                    </span>
                  </button>
                );
              })}
            </div>
          ) : (
            parsedTypedTags.length === 0 && (
              <div className="py-2 text-center text-[11px] text-slate-400">
                No tags in this dataset yet — type a tag name above and press
                Enter.
              </div>
            )
          )}
        </PopoverContent>
      </Popover>
    </div>
  );
}
