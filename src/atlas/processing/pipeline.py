from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunking import chunk_units
from .extractors import SUPPORTED_EXTENSIONS, extract_file
from .models import DocumentChunk, ProcessingResult


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inventory(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}

    inventory_path = Path(path)
    if not inventory_path.exists():
        raise FileNotFoundError(f"No existe el inventario: {inventory_path}")

    with inventory_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return {
            row["file_name"].strip(): {
                key: (value or "").strip()
                for key, value in row.items()
                if key and key != "file_name"
            }
            for row in reader
            if row.get("file_name")
        }


def _base_metadata(path: Path, inventory: dict[str, dict[str, str]]) -> dict[str, Any]:
    stat = path.stat()
    metadata: dict[str, Any] = {
        "source_file": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "file_size_bytes": stat.st_size,
        "last_modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": _sha256_file(path),
    }
    metadata.update(inventory.get(path.name, {}))
    return metadata


def _chunk_id(path: Path, metadata: dict[str, Any], index: int, text: str) -> str:
    location = json.dumps(metadata, sort_keys=True, ensure_ascii=False, default=str)
    payload = f"{path.name}|{location}|{index}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def process_document(
    path: str | Path,
    *,
    inventory_path: str | Path | None = None,
    inventory: dict[str, dict[str, str]] | None = None,
    max_chars: int = 1200,
    overlap_chars: int = 180,
) -> ProcessingResult:
    source = Path(path)
    inventory_data = inventory if inventory is not None else load_inventory(inventory_path)
    units, warnings = extract_file(source)
    chunked_units = chunk_units(
        units,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    base = _base_metadata(source, inventory_data)

    chunks: list[DocumentChunk] = []
    for index, unit in enumerate(chunked_units, start=1):
        metadata = {**base, **unit.metadata, "chunk_index": index}
        chunks.append(
            DocumentChunk(
                chunk_id=_chunk_id(source, metadata, index, unit.text),
                text=unit.text,
                metadata=metadata,
            )
        )

    if not chunks:
        warnings.append("El archivo no produjo chunks utilizables.")

    return ProcessingResult(
        source_file=source.name,
        file_type=source.suffix.lower().lstrip("."),
        chunks=chunks,
        warnings=warnings,
    )


def process_directory(
    input_dir: str | Path,
    *,
    output_jsonl: str | Path,
    inventory_path: str | Path | None = None,
    max_chars: int = 1200,
    overlap_chars: int = 180,
) -> dict[str, Any]:
    source_dir = Path(input_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"No existe el directorio: {source_dir}")

    inventory = load_inventory(inventory_path)
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No hay documentos compatibles en {source_dir}")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(source_dir),
        "output_jsonl": str(output_path),
        "documents": [],
        "errors": [],
        "total_chunks": 0,
    }

    with output_path.open("w", encoding="utf-8") as output:
        for path in files:
            try:
                result = process_document(
                    path,
                    inventory=inventory,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                )
                for chunk in result.chunks:
                    output.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                report["documents"].append(
                    {
                        "source_file": result.source_file,
                        "file_type": result.file_type,
                        "chunk_count": len(result.chunks),
                        "warnings": result.warnings,
                    }
                )
                report["total_chunks"] += len(result.chunks)
            except Exception as error:  # noqa: BLE001 - report per-file failures
                report["errors"].append(
                    {"source_file": path.name, "error": str(error)}
                )

    return report
