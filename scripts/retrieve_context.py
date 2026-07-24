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
from atlas.indexing.pipeline import validate_manifest  # noqa: E402
from atlas.indexing.store import ChromaVectorStore  # noqa: E402
from atlas.rag import (  # noqa: E402
    DEFAULT_CROSS_ENCODER_MODEL,
    RAGRetriever,
    RetrievalConfig,
    RetrievalFilters,
    create_reranker,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recupera, reclasifica y ensambla contexto RAG para Atlas."
    )
    parser.add_argument("query", help="Pregunta del colaborador.")
    parser.add_argument("--db-path", default="data/vector-store/chroma")
    parser.add_argument("--collection", default="atlas_documents")
    parser.add_argument(
        "--provider", choices=["fastembed", "hashing"], default="fastembed"
    )
    parser.add_argument("--model", default=DEFAULT_FASTEMBED_MODEL)
    parser.add_argument("--cache-dir", default=".cache/fastembed")
    parser.add_argument("--manifest", default="data/vector-store/index-manifest.json")
    parser.add_argument(
        "--reranker", choices=["hybrid", "fastembed", "none"], default="hybrid"
    )
    parser.add_argument("--reranker-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--reranker-cache-dir", default=".cache/fastembed-reranker")
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--final-k", type=int, default=4)
    parser.add_argument("--min-vector-score", type=float, default=0.0)
    parser.add_argument("--min-rerank-score", type=float, default=0.0)
    parser.add_argument("--max-chunks-per-source", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--category")
    parser.add_argument("--file-type")
    parser.add_argument("--source-file")
    parser.add_argument("--status", default="vigente")
    parser.add_argument("--version")
    parser.add_argument("--owner")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        embedding_provider = create_embedding_provider(
            args.provider,
            model_name=args.model,
            cache_dir=args.cache_dir,
        )
        validate_manifest(args.manifest, embedding_provider)
        vector_store = ChromaVectorStore(
            args.db_path,
            collection_name=args.collection,
        )
        reranker = create_reranker(
            args.reranker,
            model_name=args.reranker_model,
            cache_dir=args.reranker_cache_dir,
        )
        retriever = RAGRetriever(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            reranker=reranker,
        )
        result = retriever.retrieve(
            args.query,
            filters=RetrievalFilters(
                category=args.category,
                file_type=args.file_type,
                source_file=args.source_file,
                status=args.status,
                version=args.version,
                owner=args.owner,
            ),
            config=RetrievalConfig(
                candidate_k=args.candidate_k,
                final_k=args.final_k,
                min_vector_score=args.min_vector_score,
                min_rerank_score=args.min_rerank_score,
                max_chunks_per_source=args.max_chunks_per_source,
                max_context_chars=args.max_context_chars,
            ),
        )
    except Exception as error:  # noqa: BLE001 - friendly CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        f"Candidatos vectoriales: {result.candidate_count} | "
        f"Fragmentos finales: {len(result.chunks)} | "
        f"Reranker: {result.reranker_provider}/{result.reranker_model}"
    )
    for chunk in result.chunks:
        lexical = (
            f" lexical={chunk.lexical_score:.4f}"
            if chunk.lexical_score is not None
            else ""
        )
        print(
            f"\n[{chunk.rank}] vector={chunk.vector_score:.4f} "
            f"rerank={chunk.rerank_score:.4f}{lexical}"
        )
        print(chunk.citation)
        print(f"Categoría: {chunk.metadata.get('category', 'sin categoría')}")
        print(chunk.text)

    if result.warnings:
        print("\nAdvertencias:")
        for warning in result.warnings:
            print(f"- {warning}")

    print("\nCONTEXTO RAG:\n")
    print(result.context)
    print("\nJSON:\n")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
