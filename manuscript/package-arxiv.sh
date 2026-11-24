#!/usr/bin/env bash
# package-arxiv.sh
set -euo pipefail

if [[ $# -ne 1 || -z "${1:-}" ]]; then
  echo "Usage: $0 <stem>" >&2
  exit 1
fi

STEM="$1"
TEX_FILE="${STEM}.tex"
BBL_FILE="${STEM}.bbl"
ARCHIVE_FILE="${STEM}.tar.gz"

if [[ ! -f "$TEX_FILE" ]]; then
  echo "Error: Required TeX file '$TEX_FILE' does not exist." >&2
  exit 1
fi

# The manuscript cites with CSL (citeproc), so the reference list is written into the .tex
# and there is normally no .bbl. A .bbl is included only if one exists (natbib/biblatex builds).
EXTRA_FILES=()
if [[ -f "$BBL_FILE" ]]; then
  EXTRA_FILES+=("$BBL_FILE")
fi

# quarto render "${STEM}.qmd" --to pdf

grep -oP '\\includegraphics(\[[^]]*\])?\{\K[^}]+' "$TEX_FILE" \
  | sort -u > figures.txt

# add extensions where LaTeX inferred them
: > figures-resolved.txt
while read -r f; do
  if [[ -f "$f" ]]; then
    echo "$f" >> figures-resolved.txt
  else
    resolved=""
    for ext in png pdf jpg jpeg; do
      if [[ -f "$f.$ext" ]]; then
        resolved="$f.$ext"
        break
      fi
    done
    if [[ -z "$resolved" ]]; then
      echo "Error: Could not resolve figure '$f' referenced in '$TEX_FILE'." >&2
      exit 1
    fi
    echo "$resolved" >> figures-resolved.txt
  fi
done < figures.txt

tar czf "$ARCHIVE_FILE" \
  "$TEX_FILE" \
  "${EXTRA_FILES[@]}" \
  -T figures-resolved.txt

echo "--- contents ($ARCHIVE_FILE) ---"
tar tzf "$ARCHIVE_FILE"
echo "Output file: $ARCHIVE_FILE"
