# chatgpt-file-splitter

A small CLI for slicing large plain-text, Markdown, CSV, and JSON files into ChatGPT-friendly chunks. Each chunk respects both conservative byte limits and token limits so you can reliably upload the pieces to ChatGPT without breaking file validity or context flow.

## Why this exists
- ChatGPT currently caps uploads at ~512 MB per file, ~2M tokens per text file, and ~50 MB for CSV/spreadsheet uploads (based on the 2025 File Uploads FAQ).
- ChatGPT context windows top out well below those maxima (~128k tokens for most models, ~110k tokens for raw document stuffing), so naive splits are still too large.
- This tool keeps every chunk well under configurable limits (defaults: 80k tokens / 100 MB text, 45 MB CSV) and preserves structure: no half CSV rows, no broken JSON objects, and no mid-sentence splits unless absolutely necessary.

## Key ChatGPT upload constraints (April 2025)
- **Max file size:** ~512 MB for any file type.
- **Text/document token cap:** ~2,000,000 tokens per text, docx, PDF, MD, JSON, etc.
- **CSV / spreadsheet size:** ~50 MB.
- **Image uploads:** ~20 MB per image (included for completeness, not processed here).
- **Per-user storage:** ~10 GB per user, ~100 GB per organization.
- **Upload rate hints:** Free ≈ 3 uploads/day, Plus/Pro/Team/Business ≈ 80 uploads per 3 hours.

_Those limits shift over time — override the CLI flags when they do._

## Project layout
```
./
├── input/   # put your giant source files here (recommended)
├── output/  # splitter writes chunks here
├── split_for_chatgpt.py
├── chunkers.py
├── utils.py
├── requirements.txt
└── README.md
```

You can point the CLI at files anywhere, but using `input/` → `output/` keeps things tidy.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Both `tiktoken` and `tqdm` are optional at runtime; if `tiktoken` is missing the splitter falls back to a character-based heuristic, and if `tqdm` is missing you simply won’t see progress bars.

## Usage
Basic syntax:
```bash
python split_for_chatgpt.py <path-to-input> [--output-dir output] \
    [--max-tokens-per-chunk 80000] [--max-bytes-per-chunk 104857600]
```
If you omit `<path-to-input>`, the CLI scans `./input/` and processes every regular file it finds, writing chunks to `./output/` (or the directory you pass with `--output-dir`). This makes it easy to drop multiple files into `input/` and run the splitter once:
```bash
python split_for_chatgpt.py  # processes everything under ./input
```

### Example 1 – Markdown / text
```bash
# Input: ./input/research_notes.md (350k words)
python split_for_chatgpt.py input/research_notes.md --output-dir output
# Output files:
#   output/research_notes.part001.md
#   output/research_notes.part002.md
#   ... (stable part numbering: partNNN)
```
Paragraphs and sentences are kept intact whenever possible, so section context stays readable.

### Example 2 – CSV (header preserved)
```bash
python split_for_chatgpt.py input/customer_dump.csv --output-dir output \
    --max-bytes-per-chunk 45000000
# Each chunk still contains the CSV header row and full records only.
```

### Example 3 – JSON array of objects / large JSON objects
```bash
python split_for_chatgpt.py input/big_payload.json --output-dir output
# Outputs:
#   output/big_payload.part001.json
#   output/big_payload.part002.json
# Each chunk is a valid JSON array containing a subset of the original objects.
```
For JSON Lines (`.jsonl`) files, the tool keeps one object per line and validates each object before writing a chunk. If you supply a large JSON object that contains an array field (e.g., `{ "records": [...] }`), the splitter duplicates lightweight metadata into each chunk and divides the array entries across chunks so every output file stays valid. When the metadata itself is very large (e.g., a huge base64 string), the splitter writes those shared fields once to `*.shared.json` and adds a pointer inside every chunk file so nothing is lost. Objects without array fields are chunked by key/value pairs (the output files are smaller objects that together contain every entry).

### Chunk naming pattern
Chunks always live in the output folder and follow `<original>.partNNN<ext>`. For example, splitting `./input/log.txt` yields `output/log.part001.txt`, `output/log.part002.txt`, etc., which makes reassembly or sequential uploads trivial.

## How chunks stay ChatGPT-safe
1. Dual guardrails: every chunk must satisfy both `--max-tokens-per-chunk` (default 80k) and `--max-bytes-per-chunk` (100 MB text, 45 MB CSV, 100 MB JSON). You can override both at runtime.
2. Structural integrity:
   - Text/Markdown: prefer paragraph boundaries, then sentences, finally byte-sized fallbacks as a last resort.
   - CSV/TSV: header row is copied into each chunk; rows are never split.
   - JSON arrays / JSONL: full objects only. Other JSON structures are either copied intact or rejected with a clear error.
3. Token counting: uses `tiktoken` when available (cl100k_base); otherwise approximates tokens as `len(text)/4`.
4. Progressive streaming: files are read paragraph/row/object by paragraph/row/object to avoid loading huge files entirely into memory. Optional `tqdm` progress bars show you how far along the read is.

## Limits change — tweak the flags
If OpenAI raises or lowers limits, adjust:
- `--max-tokens-per-chunk`: set to the new “comfortable” context window (e.g., 60k or 120k).
- `--max-bytes-per-chunk`: set to whatever margin you need under the upload caps (e.g., new CSV limit).
These defaults are conservative snapshots of today’s guidance, not hard-coded ceilings.

## Extra tips
- Use the `--plan` flag (`free`, `plus`, `pro`, `team`, `enterprise`) to print quick reminders about upload cadence and storage expectations.
- Place gigantic inputs under `./input/` and keep the generated pieces in `./output/` so you can zip and upload them later.
- Inspect the log after a run: each chunk printout includes the chunk number, size, estimated tokens, and record counts (paragraphs/rows/objects) so you know what went where.

Happy splitting!
