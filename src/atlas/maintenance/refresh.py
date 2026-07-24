from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas.indexing.embeddings import DEFAULT_FASTEMBED_MODEL, create_embedding_provider
from atlas.indexing.pipeline import index_chunks
from atlas.indexing.store import ChromaVectorStore
from atlas.processing.pipeline import process_directory


def _collection_name(prefix: str = "atlas_documents") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    return f"{prefix}_{timestamp}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_knowledge_base(
    *,
    input_dir: str | Path = "knowledge-base/documents",
    inventory_path: str | Path = "knowledge-base/document-inventory.csv",
    chunks_path: str | Path = "data/processed/chunks.jsonl",
    processing_report_path: str | Path = "data/processed/processing-report.json",
    db_path: str | Path = "data/vector-store/chroma",
    manifest_path: str | Path = "data/vector-store/index-manifest.json",
    maintenance_report_path: str | Path = "data/processed/maintenance-report.json",
    embedding_provider_name: str = "fastembed",
    embedding_model: str = DEFAULT_FASTEMBED_MODEL,
    embedding_cache_dir: str | Path = ".cache/fastembed",
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Process all source documents and atomically publish a new vector collection.

    The previous manifest remains active until the staging collection is fully indexed.
    """

    chunks = Path(chunks_path)
    processing_report_file = Path(processing_report_path)
    final_manifest = Path(manifest_path)
    maintenance_report_file = Path(maintenance_report_path)

    processing_report = process_directory(
        input_dir,
        output_jsonl=chunks,
        inventory_path=inventory_path,
    )
    processing_report_file.parent.mkdir(parents=True, exist_ok=True)
    processing_report_file.write_text(
        json.dumps(processing_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if processing_report["errors"] and not allow_partial:
        raise RuntimeError(
            "La actualización fue cancelada porque uno o más documentos fallaron. "
            f"Revisa {processing_report_file}."
        )

    previous_manifest = _read_json(final_manifest)
    staging_collection = _collection_name()
    provider = create_embedding_provider(
        embedding_provider_name,
        model_name=embedding_model,
        cache_dir=str(embedding_cache_dir),
    )
    store = ChromaVectorStore(db_path, collection_name=staging_collection)
    next_manifest = final_manifest.with_name(final_manifest.name + ".next")

    indexing_report = index_chunks(
        chunks,
        embedding_provider=provider,
        vector_store=store,
        reset=True,
        manifest_path=next_manifest,
    )
    if store.count() != indexing_report.indexed_chunks:
        raise RuntimeError("La colección de staging no contiene todos los chunks esperados.")

    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(next_manifest, final_manifest)

    report = {
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "published",
        "documents_processed": len(processing_report["documents"]),
        "processing_errors": processing_report["errors"],
        "chunks_indexed": indexing_report.indexed_chunks,
        "previous_collection": previous_manifest.get("collection_name"),
        "active_collection": staging_collection,
        "manifest_path": str(final_manifest),
        "note": (
            "La colección anterior se conserva para rollback manual. "
            "Puede eliminarse después de validar la nueva versión."
        ),
    }
    maintenance_report_file.parent.mkdir(parents=True, exist_ok=True)
    maintenance_report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
