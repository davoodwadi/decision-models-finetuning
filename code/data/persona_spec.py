"""Fail-fast loader and standardizer for global persona specifications (`persona_spec.yaml`)."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_SPEC_PATH = Path(__file__).with_name("persona_spec.yaml")
ALLOWED_SPEC_KEYS = {"prompt", "levels", "numeric_bins", "aliases", "missing", "overrides", "multi_select"}


@dataclass(frozen=True)
class NumericBin:
    """Inclusive numeric range mapped to a canonical string level."""

    min: float
    max: float
    label: str

    def __post_init__(self) -> None:
        if isinstance(self.min, bool) or isinstance(self.max, bool):
            raise TypeError(f"NumericBin bounds must be numbers, not bool: ({self.min!r}, {self.max!r})")
        if self.min > self.max:
            raise ValueError(f"NumericBin min ({self.min}) > max ({self.max}) for label {self.label!r}")
        if not isinstance(self.label, str):
            raise TypeError(f"NumericBin label must be a string, got {self.label!r}")


@dataclass(frozen=True)
class PersonaAttribute:
    """Validated specification for one global persona attribute."""

    name: str
    prompt: str
    levels: tuple[str, ...]
    numeric_bins: tuple[NumericBin, ...] = ()
    aliases: dict[object, str] = field(default_factory=dict)
    missing: tuple[object, ...] = ()
    overrides: dict[str, dict[object, str | None]] = field(default_factory=dict)
    multi_select: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.multi_select, bool):
            raise TypeError(f"{self.name}: multi_select must be bool, got {self.multi_select!r}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise TypeError(f"{self.name}: prompt must be a non-empty string, got {self.prompt!r}")

        for lvl in self.levels:
            if not isinstance(lvl, str):
                raise TypeError(f"{self.name}: level {lvl!r} must be a string (quote values like 'Yes'/'No' in YAML)")

        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.name}: duplicate entries in levels")

        allowed = set(self.levels)
        missing_set = set(self.missing)

        for b in self.numeric_bins:
            if b.label not in allowed:
                raise ValueError(f"{self.name}: numeric_bins label {b.label!r} not in levels")
        sorted_bins = sorted(self.numeric_bins, key=lambda b: (b.min, b.max))
        for prev, curr in zip(sorted_bins, sorted_bins[1:]):
            if curr.min <= prev.max:
                raise ValueError(f"{self.name}: overlapping numeric_bins {prev} and {curr}")

        overlap_levels = allowed & missing_set
        if overlap_levels:
            raise ValueError(f"{self.name}: values in both levels and missing: {sorted(overlap_levels)}")

        overlap_aliases = set(self.aliases) & missing_set
        if overlap_aliases:
            raise ValueError(f"{self.name}: values in both aliases and missing: {sorted(overlap_aliases)}")

        bad_alias_targets = set(self.aliases.values()) - allowed
        if bad_alias_targets:
            raise ValueError(f"{self.name}: alias targets not in levels: {sorted(bad_alias_targets)}")

        for dataset, mapping in self.overrides.items():
            bad_override_targets = {v for v in mapping.values() if v is not None} - allowed
            if bad_override_targets:
                raise ValueError(
                    f"{self.name}: override targets for {dataset!r} not in levels: {sorted(bad_override_targets)}"
                )


def load_persona_spec(spec_path: Path = DEFAULT_SPEC_PATH) -> dict[str, PersonaAttribute]:
    """Load and validate all persona attribute definitions from a YAML spec file."""
    with open(spec_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{spec_path}: expected top-level mapping of attribute names")

    specs: dict[str, PersonaAttribute] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"{spec_path}: attribute {name!r} must be a mapping")
        unknown_keys = set(cfg) - ALLOWED_SPEC_KEYS
        if unknown_keys:
            raise ValueError(f"{name}: unknown spec keys {sorted(unknown_keys)}")

        bins = tuple(
            NumericBin(min=b["min"], max=b["max"], label=b["label"])
            for b in (cfg["numeric_bins"] if "numeric_bins" in cfg else ())
        )

        specs[name] = PersonaAttribute(
            name=name,
            prompt=cfg["prompt"],
            levels=tuple(cfg["levels"]),
            numeric_bins=bins,
            aliases=dict(cfg["aliases"]) if "aliases" in cfg else {},
            missing=tuple(cfg["missing"]) if "missing" in cfg else (),
            overrides={ds: dict(m) for ds, m in cfg["overrides"].items()} if "overrides" in cfg else {},
            multi_select=cfg["multi_select"] if "multi_select" in cfg else False,
        )

    return specs


def standardize_column(
    series: pd.Series,
    attr: str,
    dataset: str,
    specs: dict[str, PersonaAttribute] | None = None,
) -> pd.Series:
    """Standardize one raw persona column into the global levels for `attr`."""
    all_specs = specs if specs is not None else load_persona_spec()
    spec = all_specs[attr]
    allowed = set(spec.levels)
    missing = set(spec.missing)
    overrides = spec.overrides[dataset] if dataset in spec.overrides else {}

    out: list[object] = []
    for val in series:
        if pd.isna(val):
            out.append(float("nan"))
        elif val in overrides:
            target = overrides[val]
            out.append(float("nan") if target is None else target)
        elif val in missing:
            out.append(float("nan"))
        elif val in spec.aliases:
            out.append(spec.aliases[val])
        elif val in allowed:
            out.append(val)
        elif spec.numeric_bins and isinstance(val, (int, float)) and not isinstance(val, bool):
            matched = [b.label for b in spec.numeric_bins if b.min <= val <= b.max]
            if len(matched) != 1:
                raise ValueError(f"{dataset}.{attr}: numeric value {val!r} outside numeric_bins")
            out.append(matched[0])
        elif spec.multi_select and isinstance(val, str) and "; " in val:
            tokens: set[str] = set()
            for part in val.split("; "):
                if part in overrides:
                    target = overrides[part]
                    if target is not None:
                        tokens.add(target)
                elif part in missing:
                    continue
                elif part in spec.aliases:
                    tokens.add(spec.aliases[part])
                elif part in allowed:
                    tokens.add(part)
                else:
                    raise ValueError(f"{dataset}.{attr}: unmapped multi-select token {part!r} in {val!r}")
            out.append("; ".join(sorted(tokens)) if tokens else float("nan"))
        else:
            raise ValueError(f"{dataset}.{attr}: unmapped raw value {val!r}")

    return pd.Series(out, index=series.index, name=attr, dtype=object)


def standardize_dataframe(
    df: pd.DataFrame,
    dataset: str,
    attrs: list[str] | None = None,
    specs: dict[str, PersonaAttribute] | None = None,
) -> pd.DataFrame:
    """Standardize all configured persona columns present in `df`."""
    all_specs = specs if specs is not None else load_persona_spec()
    target_attrs = attrs if attrs is not None else [c for c in df.columns if c in all_specs]
    result = df.copy()
    for attr in target_attrs:
        if attr not in result.columns:
            raise KeyError(f"{dataset}: missing expected persona column {attr!r}")
        result[attr] = standardize_column(result[attr], attr=attr, dataset=dataset, specs=all_specs)
    return result
