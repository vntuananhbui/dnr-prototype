#!/usr/bin/env bash
# Mở app demo: http://127.0.0.1:${PORT:-5001}/?seed=7
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
if [ ! -f artifacts/divide_refine_seed0.pt ] || [ ! -f artifacts/baseline_seed0.pt ]; then
  echo "Chưa có model trong artifacts/. Chạy ./reproduce.sh trước." >&2
  exit 1
fi
exec "$PY" app.py
