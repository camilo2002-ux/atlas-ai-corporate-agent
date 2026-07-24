from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.maintenance import refresh_knowledge_base  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Procesa documentos y publica una nueva colección vectorial de Atlas."
    )
    parser.add_argument("--input", default="knowledge-base/documents")
    parser.add_argument("--inventory", default="knowledge-base/document-inventory.csv")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--db-path", default="data/vector-store/chroma")
    parser.add_argument("--manifest", default="data/vector-store/index-manifest.json")
    parser.add_argument("--allow-partial", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = refresh_knowledge_base(
            input_dir=args.input,
            inventory_path=args.inventory,
            chunks_path=args.chunks,
            db_path=args.db_path,
            manifest_path=args.manifest,
            allow_partial=args.allow_partial,
        )
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
