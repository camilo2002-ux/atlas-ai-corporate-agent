from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.indexing.embeddings import (  # noqa: E402
    DEFAULT_FASTEMBED_MODEL,
    create_embedding_provider,
)
from atlas.indexing.pipeline import (  # noqa: E402
    build_where_filter,
    search_index,
    validate_manifest,
)
from atlas.indexing.store import ChromaVectorStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Busca fragmentos semánticamente similares en Atlas."
    )
    parser.add_argument("query", help="Pregunta o consulta del colaborador.")
    parser.add_argument("--db-path", default="data/vector-store/chroma")
    parser.add_argument("--collection", default="atlas_documents")
    parser.add_argument(
        "--provider",
        choices=["fastembed", "hashing"],
        default="fastembed",
    )
    parser.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser.add_argument("--cache-dir", default=".cache/fastembed")
    parser.add_argument("--manifest", default="data/vector-store/index-manifest.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category")
    parser.add_argument("--file-type")
    parser.add_argument("--source-file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        provider = create_embedding_provider(
            args.provider,
            model_name=args.model,
            cache_dir=args.cache_dir,
        )
        validate_manifest(args.manifest, provider)
        store = ChromaVectorStore(
            args.db_path,
            collection_name=args.collection,
        )
        where = build_where_filter(
            category=args.category,
            file_type=args.file_type,
            source_file=args.source_file,
        )
        results = search_index(
            args.query,
            embedding_provider=provider,
            vector_store=store,
            top_k=args.top_k,
            where=where,
        )
    except Exception as error:  # noqa: BLE001 - friendly CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not results:
        print("No se encontraron fragmentos para los filtros indicados.")
        return 0

    for position, result in enumerate(results, start=1):
        metadata = result.metadata
        location_parts = []
        for key in ("page", "section", "sheet", "row", "slide", "json_path"):
            if key in metadata:
                location_parts.append(f"{key}={metadata[key]}")
        location = ", ".join(location_parts) or "sin ubicación específica"
        print(f"\n[{position}] score={result.score:.4f}" if result.score is not None else f"\n[{position}]")
        print(f"Fuente: {metadata.get('source_file', 'desconocida')} ({location})")
        print(f"Categoría: {metadata.get('category', 'sin categoría')}")
        print(result.text)

    print("\nJSON:")
    print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
