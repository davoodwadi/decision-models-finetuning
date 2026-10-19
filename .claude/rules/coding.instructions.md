---
description: "Use when writing, modifying, or reviewing Python code or Jupyter notebooks. Enforces fail-fast behavior, strict expectations, and no silent bugs or fallbacks."
applyTo: "code/**/*.py, code/**/*.ipynb"
paths:
  - "code/**/*.py"
  - "code/**/*.ipynb"
---

# Fail-Fast Coding Guidelines

Never write code that can fail silently and introduce silent bugs. Explicitly allow code to break and raise errors immediately when inputs, state, or data do not match expectations.

## Rules

- **Require expected dictionary keys directly**: Index dictionaries with `data["key"]` rather than `data.get("key")` when the key is expected to exist. Never silently return `None` (or a fallback default) and continue execution when required data is missing.
- **Require expected attributes directly**: Access `obj.attr` directly instead of `getattr(obj, "attr", None)` when the attribute is part of the expected interface.
- **Do not swallow exceptions**: Never use bare `except:`, `except Exception: pass`, or catch-and-return-`None` patterns that hide errors. Let exceptions propagate unless there is a specific, intentional recovery requirement.
- **No silent fallback values for required state**: Avoid `value = x or fallback` or default parameter values that mask missing or invalid inputs in data pipelines, model configurations, or training/evaluation code.
- **Validate assumptions explicitly**: When tensor shapes, column names, label sets, or configuration values must satisfy invariants, assert or raise an explicit exception (`KeyError`, `ValueError`, `TypeError`, `AssertionError`) rather than skipping or coercing invalid entries.

## Examples

```python
# Good: fails immediately with KeyError if "label" is missing
label = record["label"]

# Bad: silently produces None and corrupts downstream logic
label = record.get("label")
```

```python
# Good: explicit validation of expected state
if logits.shape[-1] != num_options:
    raise ValueError(f"Expected {num_options} option logits, got shape {logits.shape}")

# Bad: silently slicing or ignoring mismatched dimensions
logits = logits[..., :num_options]
```
