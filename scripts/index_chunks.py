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
from atlas.indexing.pipeline import index_chunks  # noqa: E402
from atlas.indexing.store import ChromaVectorStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera embeddings e indexa chunks de Atlas en Chroma."
    )
    parser.add_argument(
        "--input",
        default="data/processed/chunks.jsonl",
        help="Archivo JSONL producido por la etapa de extracción.",
    )
    parser.add_argument(
        "--db-path",
        default="data/vector-store/chroma",
        help="Directorio persistente de Chroma.",
    )
    parser.add_argument("--collection", default="atlas_documents")
    parser.add_argument(
        "--provider",
        choices=["fastembed", "hashing"],
        default="fastembed",
        help="Usa hashing solo para pruebas offline; fastembed es el modelo real.",
    )
    parser.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser.add_argument("--cache-dir", default=".cache/fastembed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--manifest",
        default="data/vector-store/index-manifest.json",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina la colección antes de volver a indexar.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        provider = create_embedding_provider(
            args.provider,
            model_name=args.model,
            cache_dir=args.cache_dir,
        )
        store = ChromaVectorStore(
            args.db_path,
            collection_name=args.collection,
        )
        report = index_chunks(
            args.input,
            embedding_provider=provider,
            vector_store=store,
            batch_size=args.batch_size,
            reset=args.reset,
            manifest_path=args.manifest,
        )
    except Exception as error:  # noqa: BLE001 - friendly CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    print(f"Total actual en la colección: {store.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
