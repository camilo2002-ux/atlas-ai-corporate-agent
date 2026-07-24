from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.indexing.embeddings import DEFAULT_FASTEMBED_MODEL, create_embedding_provider  # noqa: E402
from atlas.indexing.pipeline import index_chunks, validate_manifest  # noqa: E402
from atlas.indexing.store import ChromaVectorStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crea el índice demo requerido por la interfaz.")
    parser.add_argument("--input", default="tests/fixtures/demo-chunks.jsonl")
    parser.add_argument("--db-path", default="data/vector-store/chroma")
    parser.add_argument("--manifest", default="data/vector-store/index-manifest.json")
    parser.add_argument("--collection", default="atlas_documents")
    parser.add_argument("--provider", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser.add_argument("--cache-dir", default=".cache/fastembed")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    provider = create_embedding_provider(
        args.provider,
        model_name=args.model,
        cache_dir=args.cache_dir,
    )
    manifest_path = Path(args.manifest)

    if manifest_path.is_file() and not args.force:
        try:
            manifest = validate_manifest(manifest_path, provider)
            store = ChromaVectorStore(
                args.db_path,
                collection_name=str(manifest.get("collection_name") or args.collection),
            )
            if store.count() > 0:
                print(
                    json.dumps(
                        {
                            "status": "already_ready",
                            "collection_name": store.collection_name,
                            "indexed_chunks": store.count(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 0
        except Exception:
            pass

    store = ChromaVectorStore(args.db_path, collection_name=args.collection)
    report = index_chunks(
        args.input,
        embedding_provider=provider,
        vector_store=store,
        reset=True,
        manifest_path=manifest_path,
    )
    print(json.dumps({"status": "created", **report.to_dict()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
