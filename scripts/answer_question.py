from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.generation import (  # noqa: E402
    AnswerService,
    ExtractiveEvidenceProvider,
    GenerationConfig,
    OCIChatProvider,
    OCIProviderSettings,
)
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
        description="Recupera evidencia, genera y valida una respuesta de Atlas."
    )
    parser.add_argument("query", help="Pregunta del colaborador.")
    parser.add_argument("--db-path", default="data/vector-store/chroma")
    parser.add_argument("--collection", default="atlas_documents")
    parser.add_argument(
        "--embedding-provider", choices=["fastembed", "hashing"], default="fastembed"
    )
    parser.add_argument("--embedding-model", default=DEFAULT_FASTEMBED_MODEL)
    parser.add_argument("--embedding-cache-dir", default=".cache/fastembed")
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

    parser.add_argument(
        "--llm-provider",
        choices=["extractive", "oci"],
        default="extractive",
        help="extractive no consume nube; oci usa OCI Generative AI.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--min-evidence-score", type=float, default=0.20)
    parser.add_argument("--max-validation-retries", type=int, default=1)
    parser.add_argument("--json-only", action="store_true")

    parser.add_argument("--oci-compartment-id", default=os.getenv("OCI_COMPARTMENT_ID"))
    parser.add_argument("--oci-model-id", default=os.getenv("OCI_GENAI_MODEL_ID"))
    parser.add_argument(
        "--oci-auth-mode",
        choices=["config_file", "instance_principal"],
        default=os.getenv("OCI_AUTH_MODE", "config_file"),
    )
    parser.add_argument("--oci-config-file", default=os.getenv("OCI_CONFIG_FILE", "~/.oci/config"))
    parser.add_argument("--oci-profile", default=os.getenv("OCI_CONFIG_PROFILE", "DEFAULT"))
    parser.add_argument("--oci-region", default=os.getenv("OCI_REGION"))
    parser.add_argument("--oci-endpoint", default=os.getenv("OCI_GENAI_ENDPOINT"))
    return parser


def _print_human(result: object) -> None:
    print(f"Estado: {result.status}")
    print(f"Proveedor: {result.provider}/{result.model}")
    print(f"Confianza documental: {result.evidence_score:.4f}")
    print(f"Intentos de generación: {result.attempts}")
    print("\nRESPUESTA:\n")
    print(result.answer)
    if result.references:
        print("\nFUENTES VERIFICADAS:")
        for reference in result.references:
            print(f"- {reference.citation}")
    if result.validation:
        print(
            f"\nValidación: {'aprobada' if result.validation.is_valid else 'rechazada'} "
            f"(score={result.validation.score:.4f})"
        )
    if result.warnings:
        print("\nAdvertencias:")
        for warning in result.warnings:
            print(f"- {warning}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        embedding_provider = create_embedding_provider(
            args.embedding_provider,
            model_name=args.embedding_model,
            cache_dir=args.embedding_cache_dir,
        )
        validate_manifest(args.manifest, embedding_provider)
        vector_store = ChromaVectorStore(args.db_path, collection_name=args.collection)
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

        filters = RetrievalFilters(
            category=args.category,
            file_type=args.file_type,
            source_file=args.source_file,
            status=args.status,
            version=args.version,
            owner=args.owner,
        )
        retrieval_config = RetrievalConfig(
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            min_vector_score=args.min_vector_score,
            min_rerank_score=args.min_rerank_score,
            max_chunks_per_source=args.max_chunks_per_source,
            max_context_chars=args.max_context_chars,
        )
        generation_config = GenerationConfig(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            min_evidence_score=args.min_evidence_score,
            max_validation_retries=args.max_validation_retries,
        )

        if args.llm_provider == "oci":
            if not args.oci_compartment_id or not args.oci_model_id:
                raise ValueError(
                    "Para OCI define --oci-compartment-id y --oci-model-id, o las "
                    "variables OCI_COMPARTMENT_ID y OCI_GENAI_MODEL_ID."
                )
            provider = OCIChatProvider(
                OCIProviderSettings(
                    compartment_id=args.oci_compartment_id,
                    model_id=args.oci_model_id,
                    auth_mode=args.oci_auth_mode,
                    config_file=args.oci_config_file,
                    profile=args.oci_profile,
                    region=args.oci_region,
                    service_endpoint=args.oci_endpoint,
                )
            )
        else:
            preview = retriever.retrieve(
                args.query,
                filters=filters,
                config=retrieval_config,
            )
            provider = ExtractiveEvidenceProvider(preview)
            # Reuse an immutable-equivalent retriever result by running the service normally.

        service = AnswerService(retriever=retriever, provider=provider)
        result = service.answer(
            args.query,
            filters=filters,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
        )
    except Exception as error:  # noqa: BLE001 - friendly CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not args.json_only:
        _print_human(result)
        print("\nJSON:\n")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.status in {"answered", "no_evidence"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
