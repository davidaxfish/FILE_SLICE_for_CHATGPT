"""Utility helpers for chatgpt-file-splitter."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional

try:  # Optional dependency used when available.
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    tiktoken = None  # type: ignore

try:  # Optional progress helper.
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - fallback to simple prints.
    tqdm = None  # type: ignore


# Conservative defaults derived from the ChatGPT File Upload FAQ (2025).
DEFAULT_TOKEN_LIMIT = 80_000
DEFAULT_TEXT_BYTE_LIMIT = 100 * 1024 * 1024  # 100 MB chunks by default.
DEFAULT_CSV_BYTE_LIMIT = 45 * 1024 * 1024  # Keep CSV chunks safely under 50 MB.
DEFAULT_JSON_BYTE_LIMIT = 100 * 1024 * 1024

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".log"}
SUPPORTED_CSV_EXTENSIONS = {".csv", ".tsv"}
SUPPORTED_JSON_EXTENSIONS = {".json", ".jsonl"}


@dataclass
class ChunkMetadata:
    """Simple structure describing a generated chunk."""

    index: int
    path: str
    byte_size: int
    token_estimate: int
    record_count: int


class TokenEstimator:
    """Counts tokens via tiktoken when possible, else falls back to heuristics."""

    def __init__(self) -> None:
        self._encoder = None
        if tiktoken is not None:
            try:
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self._encoder = None

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        # Rough heuristic: 1 token ~= 4 characters.
        return max(1, int(len(text) / 4.0))


TOKEN_ESTIMATOR = TokenEstimator()


def estimate_tokens(text: str) -> int:
    """Convenience wrapper that delegates to the shared estimator."""

    return TOKEN_ESTIMATOR.estimate(text)


def ensure_directory(path: str) -> None:
    """Create the directory if needed (recursively)."""

    os.makedirs(path, exist_ok=True)


def chunk_filename(base_name: str, index: int, extension: str) -> str:
    """Return a deterministic chunk file name."""

    stem = os.path.splitext(base_name)[0]
    return f"{stem}.part{index:03d}{extension}"


def human_readable_bytes(num_bytes: int) -> str:
    """Return a human-friendly byte string."""

    step = 1024.0
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < step:
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} PB"


def iter_with_progress(iterable: Iterable, total: Optional[int] = None, description: str = ""):
    """Wrap an iterable with tqdm if available."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=description, unit="unit")


def measure_text_bytes(text: str, encoding: str) -> int:
    """Return the byte length of text for a given encoding."""

    return len(text.encode(encoding))
