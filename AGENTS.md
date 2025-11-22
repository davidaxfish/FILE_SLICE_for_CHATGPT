You are a senior Python engineer and tooling architect.

Goal
----
Design and implement a small, GitHub-ready CLI tool called **chatgpt-file-splitter**.

This tool takes large plain-text files (.txt, .md, .csv, .json and similar text-based formats) and splits them into multiple smaller files so that each output file is safely within **ChatGPT file upload and context limits**, and the content remains easy for ChatGPT to understand.

You must:
- Encode the current public limits from OpenAI's **File Uploads FAQ** (last updated 2025-xx; see https://help.openai.com/en/articles/8555545-file-uploads-faq) as configurable **defaults**, NOT hard assumptions, so that users can override them if limits change in the future.
- Prioritize **semantic and structural integrity**: do not split in the middle of a CSV row or a JSON object; avoid cutting sentences/paragraphs when possible.
- Produce a **minimal, clean, ready-to-publish GitHub project** with clear structure, documentation, and no external network access required.

Key Constraints to encode as defaults
-------------------------------------
From the File Uploads FAQ and related docs, reflect at least these constraints:

- Hard file size limit: about **512 MB per file** (all file types).
- Text/document token limit: around **2,000,000 tokens per text or document file** (PDF, TXT, DOCX, MD, JSON, etc.).
- CSV / spreadsheet size limit: approximately **50 MB per CSV/spreadsheet**.
- Image limit: about **20 MB per image** (for completeness, but your tool is for text only).
- Per-user storage caps (approximate): a user has about **10 GB**, and an organization about **100 GB** total.
- Upload-rate caps (for information and README notes only, not enforced by code):
  - Free users: about **3 file uploads per day**.
  - Plus / Pro / Team / Business users: about **80 file uploads per 3 hours**.

Additionally, note that:
- ChatGPT models have finite context windows (e.g., ~128k tokens), and Enterprise docs mention ~110k tokens of document text can be directly stuffed into context before search/indexing kicks in.
- Your splitter should use a **conservative chunk token limit** (e.g., default 80k tokens) that the user can change via CLI flags.

Design Requirements
-------------------
1. **Language & runtime**
   - Use **Python 3.9+**, standard library as much as possible.
   - For token counting:
     - Prefer the `tiktoken` library if available.
     - If `tiktoken` is not installed, fall back to a simple heuristic (e.g., estimate tokens as `len(text) / 4.0`).
   - The code must still work even if `tiktoken` is not installed.

2. **File formats**
   The tool must support at least:
   - `.txt` and `.md`:
     - Split on paragraph and sentence boundaries when possible.
     - Try **not** to cut in the middle of a sentence; prefer splitting at newlines and punctuation (., !, ?).
   - `.csv`:
     - Treat the **header row** as special: every output chunk must include the header.
     - Split by rows; never cut a row in half.
     - Each output chunk must remain a valid CSV.
   - `.json`:
     Handle common patterns robustly:
     - Case A: Top-level JSON array of objects `[ {...}, {...}, ... ]`
       - Create chunks as **smaller arrays** that each contain a subset of objects.
       - Each output file must be valid JSON.
     - Case B: **JSON Lines** (JSONL-style) – one JSON object per line
       - Split by complete lines/objects.
       - Each output file is a text file with one JSON object per line.
     - Case C: Other JSON structures (single large object with nested fields)
       - If safely chunking without breaking JSON validity is non-trivial:
         - Either:
           - Use a conservative strategy that copies the whole object into one file if it fits.
           - Or emit a clear error message telling the user that this structure is not supported for splitting.
   - For unknown text-like extensions, fall back to treating them as `.txt` and splitting by paragraphs.

3. **Chunking logic**
   - The splitter must respect **two sets of limits**:
     1. **File-size limit (bytes)** — default based on 512 MB and CSV 50 MB limits.
     2. **Token limit per chunk** — default around 80k tokens, but configurable.
   - The code should:
     - First, try to keep each chunk under the token limit.
     - Also keep each chunk under a configurable byte limit (e.g., default 100 MB for text, lower for CSV).
   - The limits must be exposed as CLI flags:
     - `--max-tokens-per-chunk` (int)
     - `--max-bytes-per-chunk` (int)
   - For each output chunk, print or log metadata:
     - Chunk index, byte size, estimated token count, and number of records/paragraphs.

4. **CLI interface**
   Implement a main CLI script, e.g. `split_for_chatgpt.py`, with:
   - Positional arguments:
     - `input_path`: path to a single file to split.
   - Options:
     - `--output-dir`: directory to write split files into (default: `./output`).
     - `--max-tokens-per-chunk`: override default token limit.
     - `--max-bytes-per-chunk`: override default byte limit.
     - `--encoding`: default `utf-8`, but allow override.
     - `--plan`: optional string flag (`free`, `plus`, `pro`, `team`, `enterprise`), used only to print recommended usage tips; do not hard-block anything based on this.
   - Print clear progress messages and a concise final summary.

5. **Robustness**
   - Use `tqdm` (if installed) for a simple progress bar when processing very large files. If not installed, just fall back to plain prints.
   - Handle minor errors gracefully:
     - Invalid encoding: try a fallback encoding like `utf-8-sig` or inform the user.
     - Empty files: handle without crashing, and just report that nothing needed splitting.
     - Strip trailing whitespace where appropriate but never modify the semantic content.
   - Always ensure **every output file is structurally valid** for its format:
     - Valid UTF-8.
     - Valid JSON / CSV syntax where applicable.

6. **Project structure (GitHub-ready)**
   Create a minimal repo layout:

   - `split_for_chatgpt.py`  # main CLI entry-point
   - `chunkers.py`           # optional module containing format-specific chunking logic
   - `utils.py`              # optional helpers (token counting, IO helpers)
   - `requirements.txt`      # list only non-stdlib dependencies (e.g., `tiktoken`, `tqdm`)
   - `README.md`             # clear explanation and quick-start
   - `.gitignore`            # ignore virtualenv, __pycache__, etc.

7. **README.md content**
   The README must be **simple and clear** and include:

   - What the tool does, in 2–3 sentences.
   - A short summary of ChatGPT upload constraints:
     - Max 512 MB per file.
     - Max ~2M tokens for text/doc files.
     - CSV/spreadsheets around 50 MB.
     - Upload caps: free vs paid users (3 files/day vs 80 files/3 hours, approximately).
   - Installation:
     - `python -m venv .venv && source .venv/bin/activate` (or Windows equivalent).
     - `pip install -r requirements.txt`.
   - Usage examples:
     - Splitting a large `.txt` file.
     - Splitting a `.csv` while preserving header row.
     - Splitting a JSON array of objects.
   - Explanation of how chunks are safe for ChatGPT:
     - Each chunk is below both a conservative token limit and a size limit.
     - The structure (paragraphs/rows/objects) is preserved, which helps ChatGPT understand context.
   - A short section on **limits can change**:
     - Mention that if OpenAI updates the limits, users can change `--max-tokens-per-chunk` and `--max-bytes-per-chunk`.

8. **Quality and testing**
   - Include a small `if __name__ == "__main__":` block with `argparse` wiring.
   - Provide at least one self-contained example in comments or README that demonstrates:
     - Input file.
     - CLI command.
     - Expected output file naming pattern (e.g., `myfile.part001.txt`).
   - Ensure the code is formatted (PEP 8) and reasonably commented, but avoid over-commenting.

Workflow
--------
When you respond:

1. Briefly outline the design (file structure and main components).
2. Provide the **full source code** for:
   - `split_for_chatgpt.py`
   - `chunkers.py` (if used)
   - `utils.py` (if used)
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
3. Make sure the code is **copy-paste ready** and will run without modification after installing dependencies.
4. Avoid any external API calls or network requests; the tool must work offline on local files.
