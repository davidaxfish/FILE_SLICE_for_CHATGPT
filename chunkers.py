"""Format-aware chunkers for chatgpt-file-splitter."""
from __future__ import annotations

import csv
import io
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Tuple

from utils import (
    ChunkMetadata,
    chunk_filename,
    ensure_directory,
    estimate_tokens,
    human_readable_bytes,
    iter_with_progress,
    measure_text_bytes,
)

class ChunkingError(RuntimeError):
    """Raised when a file structure cannot be safely chunked."""


def split_file(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    """Dispatch to the correct chunker based on file extension."""

    lower_ext = extension.lower()
    if lower_ext in {".txt", ".md", ".rst", ".log"}:
        return split_text_file(
            input_path,
            output_dir,
            base_name,
            extension,
            max_tokens,
            max_bytes,
            encoding,
        )
    if lower_ext in {".csv", ".tsv"}:
        return split_csv_file(
            input_path,
            output_dir,
            base_name,
            extension,
            max_tokens,
            max_bytes,
            encoding,
        )
    if lower_ext in {".json", ".jsonl"}:
        return split_json_file(
            input_path,
            output_dir,
            base_name,
            extension,
            max_tokens,
            max_bytes,
            encoding,
        )
    # Fallback: treat as text.
    return split_text_file(
        input_path,
        output_dir,
        base_name,
        extension,
        max_tokens,
        max_bytes,
        encoding,
    )


def split_text_file(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    ensure_directory(output_dir)
    chunk_units: List[str] = []
    chunk_tokens = 0
    chunk_bytes = 0
    chunk_records = 0
    chunk_index = 1
    metadata: List[ChunkMetadata] = []

    def flush_chunk() -> None:
        nonlocal chunk_units, chunk_tokens, chunk_bytes, chunk_records, chunk_index
        if not chunk_units:
            return
        text = "".join(chunk_units)
        filename = chunk_filename(base_name, chunk_index, extension)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding=encoding) as f:
            f.write(text)
        metadata.append(
            ChunkMetadata(
                index=chunk_index,
                path=path,
                byte_size=chunk_bytes,
                token_estimate=chunk_tokens,
                record_count=chunk_records,
            )
        )
        print(
            f"Wrote chunk {chunk_index}: {chunk_records} blocks, "
            f"{human_readable_bytes(chunk_bytes)}, {chunk_tokens} tokens",
        )
        chunk_index += 1
        chunk_units = []
        chunk_tokens = 0
        chunk_bytes = 0
        chunk_records = 0

    def append_unit(unit: str) -> None:
        nonlocal chunk_units, chunk_tokens, chunk_bytes, chunk_records
        unit_bytes = measure_text_bytes(unit, encoding)
        unit_tokens = estimate_tokens(unit)
        if chunk_units and (
            chunk_bytes + unit_bytes > max_bytes or chunk_tokens + unit_tokens > max_tokens
        ):
            flush_chunk()
        chunk_units.append(unit)
        chunk_bytes += unit_bytes
        chunk_tokens += unit_tokens
        chunk_records += 1
        if chunk_bytes >= max_bytes or chunk_tokens >= max_tokens:
            flush_chunk()

    with open(input_path, "r", encoding=encoding, errors="replace") as f:
        line_iter = iter_with_progress(f, description="Text lines")
        for paragraph in iter_paragraphs(line_iter):
            append_respecting_limits(
                paragraph,
                append_unit,
                max_bytes,
                max_tokens,
                encoding,
            )
    flush_chunk()
    return metadata


def iter_paragraphs(handle: Iterable[str]) -> Iterable[str]:
    buffer: List[str] = []
    for line in handle:
        buffer.append(line)
        if not line.strip():
            yield "".join(buffer)
            buffer = []
    if buffer:
        yield "".join(buffer)


def append_respecting_limits(
    text: str,
    append: Callable[[str], None],
    max_bytes: int,
    max_tokens: int,
    encoding: str,
) -> None:
    unit_bytes = measure_text_bytes(text, encoding)
    unit_tokens = estimate_tokens(text)
    if unit_bytes <= max_bytes and unit_tokens <= max_tokens:
        append(text)
        return
    sentences = split_sentences(text)
    if len(sentences) == 1:
        for forced in force_split_text(text, max_bytes, encoding):
            append_respecting_limits(forced, append, max_bytes, max_tokens, encoding)
        return
    for sentence in sentences:
        append_respecting_limits(sentence, append, max_bytes, max_tokens, encoding)


def split_sentences(text: str) -> List[str]:
    matches = list(re.finditer(r"(?<=[.!?])\s+", text))
    if not matches:
        return [text]
    idx = 0
    parts = []
    for match in matches:
        end = match.end()
        parts.append(text[idx:end])
        idx = end
    if idx < len(text):
        parts.append(text[idx:])
    return [part for part in parts if part]


def force_split_text(text: str, max_bytes: int, encoding: str) -> List[str]:
    if not text:
        return [text]
    pieces: List[str] = []
    current: List[str] = []
    current_bytes = 0
    for char in text:
        char_bytes = len(char.encode(encoding))
        if current and current_bytes + char_bytes > max_bytes:
            pieces.append("".join(current))
            current = [char]
            current_bytes = char_bytes
        else:
            current.append(char)
            current_bytes += char_bytes
        if current_bytes >= max_bytes:
            pieces.append("".join(current))
            current = []
            current_bytes = 0
    if current:
        pieces.append("".join(current))
    return pieces


def split_csv_file(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    ensure_directory(output_dir)
    metadata: List[ChunkMetadata] = []
    chunk_index = 1
    chunk_rows: List[List[str]] = []
    chunk_bytes = 0
    chunk_tokens = 0
    chunk_records = 0

    with open(input_path, "r", encoding=encoding, errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        try:
            header = next(reader)
        except StopIteration:
            print("CSV appears empty - nothing to split.")
            return []
        header_bytes, header_tokens = csv_row_size(header, dialect, encoding)

        def flush_chunk() -> None:
            nonlocal chunk_rows, chunk_bytes, chunk_tokens, chunk_records, chunk_index
            if not chunk_rows:
                return
            filename = chunk_filename(base_name, chunk_index, extension)
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding=encoding, newline="") as out:
                writer = csv.writer(out, dialect)
                writer.writerow(header)
                writer.writerows(chunk_rows)
            metadata.append(
                ChunkMetadata(
                    index=chunk_index,
                    path=path,
                    byte_size=chunk_bytes,
                    token_estimate=chunk_tokens,
                    record_count=chunk_records,
                )
            )
            print(
                f"Wrote CSV chunk {chunk_index}: {chunk_records} rows, "
                f"{human_readable_bytes(chunk_bytes)}, {chunk_tokens} tokens",
            )
            chunk_index += 1
            chunk_rows = []
            chunk_bytes = 0
            chunk_tokens = 0
            chunk_records = 0

        row_iter = iter_with_progress(reader, description="CSV rows")
        for row in row_iter:
            row_bytes, row_tokens = csv_row_size(row, dialect, encoding)
            if chunk_rows and (
                chunk_bytes + row_bytes > max_bytes or chunk_tokens + row_tokens > max_tokens
            ):
                flush_chunk()
            if not chunk_rows:
                chunk_bytes = header_bytes
                chunk_tokens = header_tokens
            if row_bytes > max_bytes or row_tokens > max_tokens:
                raise ChunkingError(
                    "Single CSV row exceeds configured limits; consider raising the limits."
                )
            chunk_rows.append(row)
            chunk_bytes += row_bytes
            chunk_tokens += row_tokens
            chunk_records += 1
            if chunk_bytes >= max_bytes or chunk_tokens >= max_tokens:
                flush_chunk()
        flush_chunk()
    return metadata


def csv_row_size(row: List[str], dialect: csv.Dialect, encoding: str) -> Tuple[int, int]:
    buffer = io.StringIO()
    writer = csv.writer(buffer, dialect)
    writer.writerow(row)
    data = buffer.getvalue()
    return measure_text_bytes(data, encoding), estimate_tokens(data)


def split_json_file(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    mode = detect_json_mode(input_path, encoding, extension)
    if mode == "jsonl":
        return split_json_lines(
            input_path,
            output_dir,
            base_name,
            extension,
            max_tokens,
            max_bytes,
            encoding,
        )
    if mode == "array":
        return split_json_array(
            input_path,
            output_dir,
            base_name,
            extension,
            max_tokens,
            max_bytes,
            encoding,
        )
    return split_json_object(
        input_path,
        output_dir,
        base_name,
        extension,
        max_tokens,
        max_bytes,
        encoding,
    )


def detect_json_mode(path: str, encoding: str, extension: str) -> str:
    if extension.lower() == ".jsonl":
        return "jsonl"
    with open(path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(8192)
    stripped = sample.lstrip()
    if not stripped:
        return "array"
    if stripped.startswith("["):
        return "array"
    if stripped.startswith("{"):
        if looks_like_json_lines(sample):
            return "jsonl"
        return "object"
    return "array"


def looks_like_json_lines(sample: str) -> bool:
    lines = [line.strip() for line in sample.splitlines() if line.strip()]
    probe = lines[:3]
    if not probe:
        return False
    for line in probe:
        try:
            json.loads(line)
        except Exception:
            return False
    return True


def split_json_lines(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    ensure_directory(output_dir)
    metadata: List[ChunkMetadata] = []
    chunk_lines: List[str] = []
    chunk_bytes = 0
    chunk_tokens = 0
    chunk_records = 0
    chunk_index = 1

    def flush_chunk() -> None:
        nonlocal chunk_lines, chunk_bytes, chunk_tokens, chunk_records, chunk_index
        if not chunk_lines:
            return
        filename = chunk_filename(base_name, chunk_index, extension)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding=encoding) as out:
            out.write("\n".join(chunk_lines))
            out.write("\n")
        metadata.append(
            ChunkMetadata(
                index=chunk_index,
                path=path,
                byte_size=chunk_bytes,
                token_estimate=chunk_tokens,
                record_count=chunk_records,
            )
        )
        print(
            f"Wrote JSONL chunk {chunk_index}: {chunk_records} lines, "
            f"{human_readable_bytes(chunk_bytes)}, {chunk_tokens} tokens",
        )
        chunk_index += 1
        chunk_lines = []
        chunk_bytes = 0
        chunk_tokens = 0
        chunk_records = 0

    with open(input_path, "r", encoding=encoding, errors="replace") as f:
        line_iter = iter_with_progress(f, description="JSONL lines")
        for raw_line in line_iter:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChunkingError(f"Invalid JSON line detected: {exc}")
            line_bytes = measure_text_bytes(line + "\n", encoding)
            line_tokens = estimate_tokens(line)
            if chunk_lines and (
                chunk_bytes + line_bytes > max_bytes or chunk_tokens + line_tokens > max_tokens
            ):
                flush_chunk()
            if line_bytes > max_bytes or line_tokens > max_tokens:
                raise ChunkingError(
                    "Single JSON object exceeds configured limits; consider raising the limits."
                )
            chunk_lines.append(line)
            chunk_bytes += line_bytes
            chunk_tokens += line_tokens
            chunk_records += 1
            if chunk_bytes >= max_bytes or chunk_tokens >= max_tokens:
                flush_chunk()
    flush_chunk()
    return metadata


def split_json_array(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    ensure_directory(output_dir)
    metadata: List[ChunkMetadata] = []
    chunk_objects: List[str] = []
    chunk_payload_bytes = 0
    chunk_tokens = 0
    chunk_records = 0
    chunk_index = 1
    open_bytes = measure_text_bytes("[\n", encoding)
    close_bytes = measure_text_bytes("\n]", encoding)
    separator_bytes = measure_text_bytes(",\n", encoding)
    overhead = open_bytes + close_bytes
    objects_found = False

    def flush_chunk() -> None:
        nonlocal chunk_objects, chunk_payload_bytes, chunk_tokens, chunk_records, chunk_index
        if not chunk_objects:
            return
        filename = chunk_filename(base_name, chunk_index, extension)
        path = os.path.join(output_dir, filename)
        payload = "[\n" + ",\n".join(chunk_objects) + "\n]"
        with open(path, "w", encoding=encoding) as out:
            out.write(payload)
        byte_size = chunk_payload_bytes + overhead
        metadata.append(
            ChunkMetadata(
                index=chunk_index,
                path=path,
                byte_size=byte_size,
                token_estimate=chunk_tokens,
                record_count=chunk_records,
            )
        )
        print(
            f"Wrote JSON array chunk {chunk_index}: {chunk_records} objects, "
            f"{human_readable_bytes(byte_size)}, "
            f"{chunk_tokens} tokens",
        )
        chunk_index += 1
        chunk_objects = []
        chunk_payload_bytes = 0
        chunk_tokens = 0
        chunk_records = 0

    with open(input_path, "r", encoding=encoding, errors="replace") as f:
        decoder = json.JSONDecoder()
        buffer = ""
        started = False
        finished = False
        while not finished:
            chunk = f.read(65536)
            if not chunk:
                finished = True
            buffer += chunk
            while True:
                if not started:
                    buffer = buffer.lstrip()
                    if not buffer:
                        break
                    if buffer[0] != "[":
                        raise ChunkingError("Expected JSON array at top level.")
                    started = True
                    buffer = buffer[1:]
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == "]":
                    finished = True
                    buffer = buffer[1:]
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if finished:
                        raise
                    break
                serialized = json.dumps(obj, ensure_ascii=False)
                obj_bytes = measure_text_bytes(serialized, encoding)
                obj_tokens = estimate_tokens(serialized)
                additional_bytes = obj_bytes if not chunk_objects else obj_bytes + separator_bytes
                prospective_bytes = chunk_payload_bytes + additional_bytes + overhead
                if chunk_objects and (
                    prospective_bytes > max_bytes or chunk_tokens + obj_tokens > max_tokens
                ):
                    flush_chunk()
                    additional_bytes = obj_bytes
                    prospective_bytes = chunk_payload_bytes + additional_bytes + overhead
                if prospective_bytes > max_bytes or obj_tokens > max_tokens:
                    raise ChunkingError(
                        "Single JSON object exceeds configured limits; consider raising the limits."
                    )
                chunk_objects.append(serialized)
                chunk_payload_bytes += additional_bytes
                chunk_tokens += obj_tokens
                chunk_records += 1
                objects_found = True
                buffer = buffer[idx:]
                continue
            if finished:
                break
    flush_chunk()
    if not metadata and not objects_found:
        # Empty JSON array; reproduce a single empty chunk that mirrors the input.
        ensure_directory(output_dir)
        filename = chunk_filename(base_name, 1, extension)
        path = os.path.join(output_dir, filename)
        with open(input_path, "r", encoding=encoding, errors="replace") as src:
            payload = src.read()
        with open(path, "w", encoding=encoding) as dst:
            dst.write(payload)
        byte_size = measure_text_bytes(payload, encoding)
        token_count = estimate_tokens(payload)
        metadata.append(
            ChunkMetadata(
                index=1,
                path=path,
                byte_size=byte_size,
                token_estimate=token_count,
                record_count=0,
            )
        )
        print("JSON array was empty; copied file without splitting.")
    return metadata


def split_json_object(
    input_path: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    with open(input_path, "r", encoding=encoding, errors="replace") as f:
        data = f.read()
    byte_size = measure_text_bytes(data, encoding)
    token_count = estimate_tokens(data)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ChunkingError(f"Invalid JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ChunkingError("Top-level JSON value is not an object; consider JSON array or JSONL.")
    if byte_size <= max_bytes and token_count <= max_tokens:
        ensure_directory(output_dir)
        filename = chunk_filename(base_name, 1, extension)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding=encoding) as out:
            out.write(data)
        metadata = [
            ChunkMetadata(
                index=1,
                path=path,
                byte_size=byte_size,
                token_estimate=token_count,
                record_count=1,
            )
        ]
        print(
            f"JSON object copied without splitting: {human_readable_bytes(byte_size)}, {token_count} tokens"
        )
        return metadata

    list_keys = [key for key, value in payload.items() if isinstance(value, list) and value]
    if list_keys:
        key_to_chunk = max(list_keys, key=lambda key: len(payload[key]))
        return chunk_object_list_field(
            payload,
            key_to_chunk,
            output_dir,
            base_name,
            extension,
            max_tokens,
            max_bytes,
            encoding,
        )

    return chunk_flat_object(
        payload,
        output_dir,
        base_name,
        extension,
        max_tokens,
        max_bytes,
        encoding,
    )


def chunk_object_list_field(
    payload: Dict[str, Any],
    list_key: str,
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    ensure_directory(output_dir)
    metadata: List[ChunkMetadata] = []
    chunk_index = 1
    shared_payload = {k: v for k, v in payload.items() if k != list_key}
    shared_info = prepare_shared_payload(
        shared_payload,
        base_name,
        extension,
        output_dir,
        encoding,
        max_bytes,
        max_tokens,
    )
    inline_shared, shared_filename, external_keys = shared_info
    list_values = payload[list_key]
    chunk_entries: List[Any] = []

    def render(entries: List[Any]) -> Tuple[str, int, int]:
        chunk_obj = dict(inline_shared)
        if shared_filename:
            chunk_obj["__shared_payload_file__"] = shared_filename
            if external_keys:
                chunk_obj["__shared_payload_fields__"] = external_keys
        chunk_obj[list_key] = entries
        serialized = json.dumps(chunk_obj, ensure_ascii=False)
        return serialized, measure_text_bytes(serialized, encoding), estimate_tokens(serialized)

    def write_chunk(entries: List[Any], serialized: str, byte_size: int, token_count: int) -> None:
        nonlocal chunk_index
        filename = chunk_filename(base_name, chunk_index, extension)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding=encoding) as out:
            out.write(serialized)
        metadata.append(
            ChunkMetadata(
                index=chunk_index,
                path=path,
                byte_size=byte_size,
                token_estimate=token_count,
                record_count=len(entries),
            )
        )
        print(
            f"Wrote JSON object chunk {chunk_index}: key '{list_key}' -> {len(entries)} entries, "
            f"{human_readable_bytes(byte_size)}, {token_count} tokens",
        )
        chunk_index += 1

    current_serialized: str | None = None
    current_bytes = 0
    current_tokens = 0

    for entry in list_values:
        chunk_entries.append(entry)
        serialized, byte_size, token_count = render(chunk_entries)
        if byte_size > max_bytes or token_count > max_tokens:
            # Flush without the last entry.
            chunk_entries.pop()
            if not chunk_entries:
                raise ChunkingError(
                    "A single array entry in the JSON object exceeds the configured chunk limits."
                )
            if current_serialized is None:
                current_serialized, current_bytes, current_tokens = render(chunk_entries)
            write_chunk(chunk_entries, current_serialized, current_bytes, current_tokens)
            chunk_entries = [entry]
            serialized, byte_size, token_count = render(chunk_entries)
            if byte_size > max_bytes or token_count > max_tokens:
                raise ChunkingError(
                    "A single array entry in the JSON object exceeds the configured chunk limits."
                )
            current_serialized = None
            current_bytes = 0
            current_tokens = 0
        current_serialized = serialized
        current_bytes = byte_size
        current_tokens = token_count

    if chunk_entries:
        if current_serialized is None:
            current_serialized, current_bytes, current_tokens = render(chunk_entries)
        write_chunk(chunk_entries, current_serialized, current_bytes, current_tokens)
        current_serialized = None
        current_bytes = 0
        current_tokens = 0

    return metadata


def prepare_shared_payload(
    shared_payload: Dict[str, Any],
    base_name: str,
    extension: str,
    output_dir: str,
    encoding: str,
    max_bytes: int,
    max_tokens: int,
) -> Tuple[Dict[str, Any], str | None, List[str]]:
    if not shared_payload:
        return {}, None, []

    # Precompute size of each field so we can move the largest ones out if needed.
    field_sizes: Dict[str, Tuple[int, int]] = {}
    for key, value in shared_payload.items():
        field_json = json.dumps({key: value}, ensure_ascii=False)
        field_sizes[key] = (measure_text_bytes(field_json, encoding), estimate_tokens(field_json))

    inline_shared = dict(shared_payload)

    def current_size() -> Tuple[int, int]:
        data = json.dumps(inline_shared, ensure_ascii=False)
        return measure_text_bytes(data, encoding), estimate_tokens(data)

    inline_bytes, inline_tokens = current_size()
    limit_bytes = max(1, int(max_bytes * 0.6))
    limit_tokens = max(1, int(max_tokens * 0.6))

    external_fields: Dict[str, Any] = {}
    while inline_shared and (inline_bytes > limit_bytes or inline_tokens > limit_tokens):
        target_key = max(inline_shared, key=lambda key: field_sizes[key][0])
        external_fields[target_key] = inline_shared.pop(target_key)
        inline_bytes, inline_tokens = current_size()

    if inline_shared and (inline_bytes > max_bytes or inline_tokens > max_tokens):
        for key in list(inline_shared.keys()):
            external_fields[key] = inline_shared.pop(key)
        inline_bytes, inline_tokens = 0, 0

    shared_filename: str | None = None
    external_keys: List[str] = []
    if external_fields:
        stem, _ = os.path.splitext(base_name)
        shared_filename = f"{stem}.shared{extension}"
        shared_path = os.path.join(output_dir, shared_filename)
        with open(shared_path, "w", encoding=encoding) as out:
            json.dump(external_fields, out, ensure_ascii=False)
        external_keys = sorted(external_fields.keys())
        print(
            f"Shared fields extracted to {shared_path}: {len(external_fields)} field(s) stored separately."
        )

    return inline_shared, shared_filename, external_keys


def chunk_flat_object(
    payload: Dict[str, Any],
    output_dir: str,
    base_name: str,
    extension: str,
    max_tokens: int,
    max_bytes: int,
    encoding: str,
) -> List[ChunkMetadata]:
    ensure_directory(output_dir)
    metadata: List[ChunkMetadata] = []
    chunk_index = 1
    chunk_obj: Dict[str, Any] = {}
    chunk_serialized = ""
    chunk_bytes = 0
    chunk_tokens = 0

    def flush() -> None:
        nonlocal chunk_obj, chunk_serialized, chunk_bytes, chunk_tokens, chunk_index
        if not chunk_obj:
            return
        filename = chunk_filename(base_name, chunk_index, extension)
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding=encoding) as out:
            out.write(chunk_serialized)
        metadata.append(
            ChunkMetadata(
                index=chunk_index,
                path=path,
                byte_size=chunk_bytes,
                token_estimate=chunk_tokens,
                record_count=len(chunk_obj),
            )
        )
        print(
            f"Wrote JSON object chunk {chunk_index}: {len(chunk_obj)} entries, "
            f"{human_readable_bytes(chunk_bytes)}, {chunk_tokens} tokens",
        )
        chunk_index += 1
        chunk_obj = {}
        chunk_serialized = ""
        chunk_bytes = 0
        chunk_tokens = 0

    for key, value in payload.items():
        chunk_obj[key] = value
        serialized = json.dumps(chunk_obj, ensure_ascii=False)
        byte_size = measure_text_bytes(serialized, encoding)
        token_count = estimate_tokens(serialized)
        if byte_size > max_bytes or token_count > max_tokens:
            chunk_obj.pop(key)
            if not chunk_obj:
                raise ChunkingError(
                    f"Key '{key}' in JSON object exceeds the configured limits on its own."
                )
            flush()
            chunk_obj[key] = value
            serialized = json.dumps(chunk_obj, ensure_ascii=False)
            byte_size = measure_text_bytes(serialized, encoding)
            token_count = estimate_tokens(serialized)
            if byte_size > max_bytes or token_count > max_tokens:
                raise ChunkingError(
                    f"Key '{key}' in JSON object exceeds the configured limits on its own."
                )
        chunk_serialized = serialized
        chunk_bytes = byte_size
        chunk_tokens = token_count

    flush()
    return metadata
