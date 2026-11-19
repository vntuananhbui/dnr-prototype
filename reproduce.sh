#!/usr/bin/env bash
# Một lệnh: dữ liệu -> train (4 cấu hình × 3 seed) -> eval -> report. Ngân sách ≤ 10 phút trên CPU.
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python3; fi
echo "python: $PY"
START=$(date +%s)
"$PY" data.py
"$PY" train.py
"$PY" eval.py
"$PY" make_report.py
test -s results/metrics.json
test -s report/index.html
echo "reproduce.sh xong trong $(( $(date +%s) - START )) giây"
