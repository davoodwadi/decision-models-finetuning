# Word allocation: Marketing Science papers in `lit/` vs. our manuscript

Counted 2026-11-22. Papers compared:

- **Leng, Sang & Agarwal (2026)**, "Frontiers: Reducing Disparity Between LLMs and Humans: Optimal LLM Sample Calibration," *Marketing Science*, Articles in Advance. Frontiers format (same 6,000-word limit as our target), 13 pages, 7 figures, 0 tables, ~28 references.
- **Wang, Zhang & Zhang (2026)**, "Large Language Models for Market Research: A Data-Augmentation Approach," *Marketing Science* 45(4):728–751. Regular article (no word limit), 24 pages, 3 figures, 14 tables, ~64 references.
- **Ours**: `manuscript-training.docx` (current draft), counted up to the "Web Appendix" heading, the same rule `count_word.py` uses. 5 figures, 2 tables, 41 references. No abstract yet, Discussion is a placeholder, and §5 still has a TODO.

The Psychology & Marketing paper in `lit/` is not a Marketing Science paper, so it is left out.

## Method

- MS papers: `pdftotext` on the published PDF. Removed running heads, the "Marketing Science, …" footer line, bare page numbers, and the INFORMS copyright notice at the end. Split at top-level numbered headings and at Abstract / Acknowledgments / Endnotes / References / author bios.
- Ours: the docx paragraphs and table cells, split at Heading 1 (the same logic as `count_word.py`; totals match its output).
- **Tokens** = whitespace-separated strings (the `count_word.py` definition). **Alpha** = tokens that contain at least one letter. Alpha drops bare numbers and most equation fragments, which matters for the math-heavy Wang paper.
- Caveats: PDF text includes words inside figures (axis labels, legends), table cells, and equation fragments. Two-column extraction can move a few lines across a section boundary. Published versions come after revision, so they are not the submitted counts.

## Leng et al. (Frontiers): section counts

| Section | Tokens | Alpha | % of body |
|---|---:|---:|---:|
| Title, authors, dates | 90 | 75 | |
| Abstract | 260 | 260 | |
| Metadata (history, funding, supplement, keywords) | 85 | 78 | |
| 1. Introduction | 828 | 793 | 14.2% |
| 2. Human Mimicry Calibration Framework (method) | 1,946 | 1,834 | 33.5% |
| 3. Revealed Preference Calibration (application 1) | 1,925 | 1,834 | 33.1% |
| 4. Opinion Calibration: Public Opinion (application 2) | 844 | 785 | 14.5% |
| 5. Conclusion | 271 | 270 | 4.7% |
| **Body (§1–5)** | **5,814** | **5,516** | 100% |
| Acknowledgments | 62 | 62 | |
| Endnotes | 533 | 509 | |
| References | 607 | 558 | |
| Author bios | 154 | 154 | |
| **Total** | **7,605** | **7,212** | |

Structure: there is no literature review section. The literature goes into a contributions paragraph near the end of the Introduction. The method section is written as a numbered seven-step road map (Figure 1), and both applications are organized by those same steps. There is a lot of endnote text (533 tokens, 7% of the paper), and appendix material is referenced throughout.

## Wang et al. (regular article): section counts

| Section | Tokens | Alpha | % of body |
|---|---:|---:|---:|
| Title, authors, dates | 116 | 96 | |
| Abstract | 245 | 244 | |
| Metadata (history, funding, supplement, keywords) | 40 | 38 | |
| 1. Introduction | 1,762 | 1,708 | 10.3% |
| 2. Literature | 1,365 | 1,317 | 8.0% |
| 3. Model and the Data-Augmentation Approach | 2,526 | 2,279 | 14.8% |
| 4. Theoretical Properties of the AAE | 3,909 | 3,436 | 22.8% |
| 5. Empirical Analysis I: COVID-19 | 4,957 | 4,423 | 29.0% |
| 6. Additional Empirical Analysis | 2,293 | 1,908 | 13.4% |
| 7. General Discussion | 303 | 301 | 1.8% |
| **Body (§1–7)** | **17,115** | **15,372** | 100% |
| Endnotes | 247 | 238 | |
| References | 1,398 | 1,262 | |
| Author bios | 197 | 192 | |
| **Total** | **19,358** | **17,442** | |

## Ours: section counts (before Web Appendix)

| Section | Tokens | Alpha | % of body |
|---|---:|---:|---:|
| Front matter (title, keywords; no abstract yet) | 21 | 21 | |
| 1. Introduction | 500 | 474 | 14.9% |
| 2. Literature review | 606 | 567 | 18.1% |
| 3. Methodology | 678 | 647 | 20.2% |
| 4. Experimental setup | 611 | 559 | 18.2% |
| 5. Results | 952 | 850 | 28.4% |
| 6. Discussion and Conclusion | 4 | 3 | 0.1% |
| **Body (§1–6)** | **3,351** | **3,100** | 100% |
| Declarations | 45 | 45 | |
| References | 1,084 | 1,006 | |
| **Total (count_word.py)** | **4,501** | **4,172** | |

## Side-by-side allocation (share of body tokens)

Sections are grouped by what they do, so the three papers can be compared.

| Function | Leng (Frontiers) | Wang (regular) | Ours (draft) |
|---|---:|---:|---:|
| Motivation + literature (intro, lit review) | 14.2% | 18.3% | 33.0% |
| Method / model / theory | 33.5% | 37.6% | 20.2% |
| Empirical (setup + results) | 47.6% | 42.4% | 46.6% |
| Conclusion / discussion | 4.7% | 1.8% | 0.1% (placeholder) |

## Share of the paper's own content (abstract + body + endnotes + references)

Title block, journal metadata, acknowledgments, and bios are left out, because they are production material, not the authors' content.

| Component | Leng | Wang | Ours |
|---|---:|---:|---:|
| Abstract | 260 (3.6%) | 245 (1.3%) | 0 (not written) |
| Body | 5,814 (80.6%) | 17,115 (90.1%) | 3,351 (75.6%) |
| Endnotes / footnotes | 533 (7.4%) | 247 (1.3%) | 0 |
| References | 607 (8.4%) | 1,398 (7.4%) | 1,084 (24.4%) |
| **Total** | **7,214** | **19,005** | **4,435** |
| References: entries / tokens per entry | ~28 / ~22 | ~64 / ~22 | 41 / ~26 |

## Differences from the Frontiers comparator (Leng et al.)

1. **Motivation + literature takes 2.3× Leng's share** (33.0% vs. 14.2%). Leng has no standalone literature review. Positioning against prior work is one contributions paragraph inside the Introduction.
2. **Method takes about 60% of Leng's share** (20.2% vs. 33.5%). Leng's method section is the largest part of the paper. Ours is shorter than our literature review plus setup.
3. **The empirical share is about the same** (46.6% vs. 47.6%). Leng splits it 2:1 between a main and a secondary application. Ours splits it into a setup section and a results section.
4. **References take 24% of our counted words vs. 8% for Leng.** We cite more entries (41 vs. ~28), and each entry is longer (~26 vs. ~22 tokens). Under a 6,000-word cap that counts references, this is the largest difference between the two papers in what the words are spent on.
5. **Leng puts 533 words in endnotes.** Endnotes count toward the Frontiers limit, so they are not a way to save words. Leng uses them for caveats and definitions that would otherwise interrupt the main text.
6. **Leng's published Frontiers paper counts above 6,000 by this method** (7,214 tokens of author content; 6,843 alpha-only). Part of that is figure-internal text and equation fragments picked up by `pdftotext`, and the count is post-revision. So treat Leng's absolute numbers as an upper bound and compare shares, not totals.
