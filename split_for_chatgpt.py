#!/usr/bin/env python3
"""CLI entry point for chatgpt-file-splitter."""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

from chunkers import ChunkingError, split_file
from utils import (
    ChunkMetadata,
    DEFAULT_CSV_BYTE_LIMIT,
    DEFAULT_JSON_BYTE_LIMIT,
    DEFAULT_TEXT_BYTE_LIMIT,
    DEFAULT_TOKEN_LIMIT,
    SUPPORTED_CSV_EXTENSIONS,
    SUPPORTED_JSON_EXTENSIONS,
    ensure_directory,
    human_readable_bytes,
)

PLAN_TIPS = {
    "free": "Free plans typically allow about 3 uploads/day. Consider batching chunks wisely.",
    "plus": "Plus users see roughly 80 uploads/3 hours. Larger batches are usually fine.",
    "pro": "Pro / Team tiers allow higher upload bursts (~80 per 3 hours).",
    "team": "Team plans often combine with shared org storage (~100 GB).",
    "enterprise": "Enterprise orgs can index ~110k tokens per document before search handoff.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split large text-like files into ChatGPT-friendly chunks that respect conservative "
            "token and byte limits."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a file or directory. If omitted, ./input is scanned for files.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where chunk files are written (default: ./output).",
    )
    parser.add_argument(
        "--max-tokens-per-chunk",
        type=int,
        default=DEFAULT_TOKEN_LIMIT,
        help="Override the token limit each chunk should respect (default: 80k).",
    )
    parser.add_argument(
        "--max-bytes-per-chunk",
        type=int,
        default=None,
        help="Override the byte-size limit per chunk (defaults vary by format).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding to use when reading/writing (default: utf-8).",
    )
    parser.add_argument(
        "--plan",
        choices=["free", "plus", "pro", "team", "enterprise"],
        help="Optional plan hint to print usage guidance.",
    )
    return parser.parse_args()


def resolve_paths(raw_path: str | None) -> List[str]:
    """Return a list of input files based on the provided path or ./input by default."""

    def normalize(path: str) -> str:
        return os.path.abspath(os.path.expanduser(path))

    def gather_from_directory(directory: str) -> List[str]:
        if not os.path.isdir(directory):
            return []
        collected = []
        for entry in sorted(os.listdir(directory)):
            full_path = os.path.join(directory, entry)
            if os.path.isfile(full_path):
                collected.append(os.path.abspath(full_path))
        return collected

    if raw_path:
        expanded = normalize(raw_path)
        if os.path.isdir(expanded):
            files = gather_from_directory(expanded)
            if not files:
                raise FileNotFoundError(f"No files found inside directory: {expanded}")
            return files
        if os.path.isfile(expanded):
            return [expanded]
        # As a second attempt, treat path as relative to ./input.
        alt = os.path.join("input", raw_path)
        if os.path.isdir(alt):
            files = gather_from_directory(alt)
            if not files:
                raise FileNotFoundError(f"No files found inside directory: {alt}")
            return files
        if os.path.isfile(alt):
            return [os.path.abspath(alt)]
        raise FileNotFoundError(f"Input not found: {raw_path}")

    default_dir = os.path.abspath("input")
    files = gather_from_directory(default_dir)
    if files:
        return files
    raise FileNotFoundError(
        "No input_path provided and ./input has no files. Supply a file or add files to ./input."
    )


def pick_default_bytes(extension: str) -> int:
    lower_ext = extension.lower()
    if lower_ext in SUPPORTED_CSV_EXTENSIONS:
        return DEFAULT_CSV_BYTE_LIMIT
    if lower_ext in SUPPORTED_JSON_EXTENSIONS:
        return DEFAULT_JSON_BYTE_LIMIT
    return DEFAULT_TEXT_BYTE_LIMIT


def summarize(metadata: List[ChunkMetadata]) -> tuple[int, int]:
    if not metadata:
        print("No chunks were emitted (input was empty or header-only).")
        return 0, 0
    total_bytes = sum(item.byte_size for item in metadata)
    total_tokens = sum(item.token_estimate for item in metadata)
    print(
        f"Finished: {len(metadata)} chunk(s), total output {human_readable_bytes(total_bytes)} "
        f"(~{total_tokens} tokens)."
    )
    return total_bytes, total_tokens


def main() -> None:
    args = parse_args()
    try:
        targets = resolve_paths(args.input_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    ensure_directory(output_dir)

    if args.plan:
        tip = PLAN_TIPS.get(args.plan.lower())
        if tip:
            print(f"Plan tip ({args.plan}): {tip}")

    overall_bytes = 0
    overall_tokens = 0
    overall_chunks = 0

    for target in targets:
        base_name = os.path.basename(target)
        _, extension = os.path.splitext(base_name)
        if not extension:
            extension = ".txt"
        max_bytes_config = args.max_bytes_per_chunk or pick_default_bytes(extension)
        max_tokens_config = args.max_tokens_per_chunk
        try:
            file_size = os.path.getsize(target)
        except OSError:
            file_size = 0
        estimated_min_chunks = max(
            1, (file_size // max_bytes_config) + (1 if file_size % max_bytes_config else 0)
        )

        print("-" * 80)
        print(f"Processing {target} -> {output_dir}")
        print(
            f"Limits per chunk: {human_readable_bytes(max_bytes_config)} and ~{max_tokens_config} tokens"
        )
        if file_size:
            print(
                f"Input size: {human_readable_bytes(file_size)} | Byte-based chunk estimate: ~{estimated_min_chunks}"
            )

        try:
            metadata = split_file(
                input_path=target,
                output_dir=output_dir,
                base_name=base_name,
                extension=extension,
                max_tokens=max_tokens_config,
                max_bytes=max_bytes_config,
                encoding=args.encoding,
            )
        except ChunkingError as exc:
            print(f"Cannot split file {target}: {exc}", file=sys.stderr)
            continue
        except Exception as exc:  # pragma: no cover - unexpected errors.
            print(f"Unexpected error on {target}: {exc}", file=sys.stderr)
            continue

        file_bytes, file_tokens = summarize(metadata)
        overall_bytes += file_bytes
        overall_tokens += file_tokens
        overall_chunks += len(metadata)

    if overall_chunks:
        print("=" * 80)
        print(
            f"All files done: {overall_chunks} total chunk(s), {human_readable_bytes(overall_bytes)} "
            f"(~{overall_tokens} tokens) written to {output_dir}"
        )
    else:
        print("No chunks were produced. Verify the input files contain supported data.")


if __name__ == "__main__":
    main()
