"use client";

import * as React from "react";
import { ChevronDownIcon, XIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

interface MultiSelectFilterProps {
  options: Array<[string, number]>;
  selected: string[];
  onChange: (next: string[]) => void;
  allLabel: string;
  singularNoun: string;
  pluralNoun: string;
  searchPlaceholder?: string;
  className?: string;
}

export function MultiSelectFilter({
  options,
  selected,
  onChange,
  allLabel,
  singularNoun,
  pluralNoun,
  searchPlaceholder = "Filter options...",
  className,
}: MultiSelectFilterProps) {
  const [open, setOpen] = React.useState(false);
  const selectedSet = React.useMemo(() => new Set(selected), [selected]);

  const toggleValue = (val: string) => {
    if (selectedSet.has(val)) {
      onChange(selected.filter((x) => x !== val));
    } else {
      onChange([...selected, val]);
    }
  };

  return (
    <div className={cn("relative flex items-center", className)}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger
          className={cn(
            buttonVariants({ variant: "outline", size: "sm" }),
            "w-full justify-between bg-white px-2.5 text-xs font-normal text-slate-700 shadow-none",
            selected.length > 0 && "pr-7 border-slate-300 bg-slate-50/80 font-medium text-slate-900"
          )}
        >
          <span className="flex items-center gap-1.5 min-w-0 truncate">
            {selected.length === 0 ? (
              <span className="truncate">{allLabel}</span>
            ) : selected.length === 1 ? (
              <span className="truncate font-mono text-[11px]">
                {selected[0]}
              </span>
            ) : (
              <>
                <Badge
                  variant="secondary"
                  className="h-4 px-1.5 text-[10px] font-mono rounded-sm"
                >
                  {selected.length}
                </Badge>
                <span className="truncate">
                  {selected.length === 1 ? singularNoun : pluralNoun}
                </span>
              </>
            )}
          </span>
          {selected.length === 0 && (
            <ChevronDownIcon className="size-3.5 text-muted-foreground shrink-0" />
          )}
        </PopoverTrigger>

        <PopoverContent align="start" className="w-64 p-0">
          <Command>
            <CommandInput
              placeholder={searchPlaceholder}
              className="text-xs"
            />
            <CommandList className="max-h-64">
              <CommandEmpty className="py-4 text-xs text-muted-foreground">
                No matching {pluralNoun}.
              </CommandEmpty>
              <CommandGroup>
                {options.map(([val, count]) => {
                  const checked = selectedSet.has(val);
                  return (
                    <CommandItem
                      key={val}
                      value={val}
                      onSelect={() => toggleValue(val)}
                      className="cursor-pointer text-xs py-1.5"
                    >
                      <Checkbox
                        checked={checked}
                        tabIndex={-1}
                        className="pointer-events-none"
                      />
                      <span className="truncate font-mono text-xs">{val}</span>
                      <CommandShortcut className="font-mono tracking-normal text-[11px]">
                        {count}
                      </CommandShortcut>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            </CommandList>
            <CommandSeparator />
            <div className="flex items-center justify-between gap-1 p-1">
              <Button
                type="button"
                variant="ghost"
                size="xs"
                className="flex-1 text-xs"
                onClick={() => onChange(options.map(([v]) => v))}
              >
                Select all
              </Button>
              {selected.length > 0 && (
                <Button
                  type="button"
                  variant="ghost"
                  size="xs"
                  className="flex-1 text-xs text-muted-foreground hover:text-foreground"
                  onClick={() => onChange([])}
                >
                  Clear ({selected.length})
                </Button>
              )}
            </div>
          </Command>
        </PopoverContent>
      </Popover>

      {selected.length > 0 && (
        <button
          type="button"
          aria-label={`Clear ${pluralNoun} filter`}
          onClick={(e) => {
            e.stopPropagation();
            onChange([]);
          }}
          className="absolute right-2 top-1/2 -translate-y-1/2 rounded p-0.5 text-slate-400 hover:bg-slate-200/70 hover:text-slate-700 cursor-pointer"
        >
          <XIcon className="size-3" />
        </button>
      )}
    </div>
  );
}
