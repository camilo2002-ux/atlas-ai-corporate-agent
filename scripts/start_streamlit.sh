#!/usr/bin/env bash
set -euo pipefail

if [[ "${ATLAS_BOOTSTRAP_DEMO:-true}" == "true" ]]; then
  python scripts/bootstrap_demo_index.py
fi

exec streamlit run streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}" \
  --server.headless=true
