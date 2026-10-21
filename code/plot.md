# Academic plotting palettes and style guide

This document defines the two standard six-color categorical palettes and supporting structural tones used for figures in this repository. Both schemes provide strong luminance separation for grayscale printing, clear contrast across multi-group bar and line charts, and restrained saturation suitable for academic manuscripts.

## Nature and NPG editorial palette

The Nature editorial palette pairs a deep Oxford navy primary with a muted terracotta secondary, followed by sage viridian, warm ochre, muted plum, and slate steel. In two-group comparisons (e.g., male versus female response distributions), use the first two colors (`#1F3B5B` and `#B84A39`). In multi-group comparisons across demographic subpopulations (e.g., age brackets, education levels, or partisan groups), cycle through the six hues in order.

| Order | Role | Color name | Hex code | RGB |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Primary group A | Oxford navy | `#1F3B5B` | `(31, 59, 91)` |
| 2 | Primary group B | Muted terracotta | `#B84A39` | `(184, 74, 57)` |
| 3 | Group C | Sage viridian | `#2A7B6F` | `(42, 123, 111)` |
| 4 | Group D | Warm ochre | `#D4912A` | `(212, 145, 42)` |
| 5 | Group E | Muted plum | `#6A4C7D` | `(106, 76, 125)` |
| 6 | Group F | Slate steel | `#5C768D` | `(92, 118, 141)` |

## Lancet and NEJM clinical palette

The Lancet clinical palette provides a slightly higher-contrast alternative anchored by dark clinical navy and carmine red, followed by teal green, muted amber, royal violet, and Nordic steel. As with the Nature palette, the first two entries (`#1B365D` and `#9E2A2B`) serve as the default pair for binary contrasts, while the full six-color sequence supports up to six demographic subpopulations.

| Order | Role | Color name | Hex code | RGB |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Primary group A | Clinical navy | `#1B365D` | `(27, 54, 93)` |
| 2 | Primary group B | Clinical carmine | `#9E2A2B` | `(158, 42, 43)` |
| 3 | Group C | Teal green | `#2E8B78` | `(46, 139, 120)` |
| 4 | Group D | Muted amber | `#D9822B` | `(217, 130, 43)` |
| 5 | Group E | Royal violet | `#6F5091` | `(111, 80, 145)` |
| 6 | Group F | Nordic steel | `#4A6F8A` | `(74, 111, 138)` |

## Neutral structural tones and Matplotlib configuration

Reference lines, baseline models, gridlines, and annotations should use warm neutral grays so that structural elements remain recessive behind data marks. Specifically, use `#6E6A64` for dashed reference lines (e.g., uniform baseline shares), `#9A9893` for baseline model bars or markers, `#52514E` for bar value labels and axis ticks, `#D4D2CC` for subtle gridlines, and `#E0DED8` for legend borders. When comparing human ground truth against off-the-shelf and fine-tuned models, map human ground truth to `#1F3B5B` (or `#1B365D` in the Lancet scheme), the off-the-shelf baseline to `#B84A39` (or `#9E2A2B`), and the fine-tuned model to `#2A7B6F` (or `#2E8B78`).

```python
import matplotlib.pyplot as plt
from cycler import cycler

PALETTES = {
    "nature": ["#1F3B5B", "#B84A39", "#2A7B6F", "#D4912A", "#6A4C7D", "#5C768D"],
    "lancet": ["#1B365D", "#9E2A2B", "#2E8B78", "#D9822B", "#6F5091", "#4A6F8A"],
}

MODEL_PALETTES = {
    "nature": {
        "human": "#1F3B5B",
        "off_the_shelf": "#B84A39",
        "fine_tuned": "#2A7B6F",
    },
    "lancet": {
        "human": "#1B365D",
        "off_the_shelf": "#9E2A2B",
        "fine_tuned": "#2E8B78",
    },
}

NEUTRALS = {
    "text": "#2B2B2B",
    "muted": "#52514E",
    "ref_line": "#6E6A64",
    "baseline": "#9A9893",
    "grid": "#D4D2CC",
    "legend_edge": "#E0DED8",
}

def set_academic_palette(name="nature"):
    if name not in PALETTES:
        raise ValueError(f"Unknown palette {name!r}, expected one of {list(PALETTES.keys())}")
    colors = PALETTES[name]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    return colors
```
