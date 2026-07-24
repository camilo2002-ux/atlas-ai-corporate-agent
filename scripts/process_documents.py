from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.processing.pipeline import process_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrae, limpia y divide documentos para Atlas."
    )
    parser.add_argument(
        "--input",
        default="knowledge-base/documents",
        help="Carpeta que contiene los documentos originales.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/chunks.jsonl",
        help="Archivo JSONL que recibirá los chunks.",
    )
    parser.add_argument(
        "--report",
        default="data/processed/processing-report.json",
        help="Archivo JSON con el resumen del procesamiento.",
    )
    parser.add_argument(
        "--inventory",
        default="knowledge-base/document-inventory.csv",
        help="Inventario CSV con metadatos corporativos.",
    )
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=180)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = process_directory(
            args.input,
            output_jsonl=args.output,
            inventory_path=args.inventory,
            max_chars=args.max_chars,
            overlap_chars=args.overlap_chars,
        )
    except Exception as error:  # noqa: BLE001 - friendly CLI boundary
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Documentos procesados: {len(report['documents'])}")
    print(f"Chunks generados: {report['total_chunks']}")
    print(f"Errores: {len(report['errors'])}")
    print(f"Salida: {args.output}")
    print(f"Reporte: {args.report}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
