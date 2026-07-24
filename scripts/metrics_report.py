from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atlas.monitoring import EventStore, summarize_events  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume las métricas de calidad de Atlas.")
    parser.add_argument("--events", default="data/feedback/events.jsonl")
    args = parser.parse_args()

    metrics = summarize_events(EventStore(args.events).iter_events())
    print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
