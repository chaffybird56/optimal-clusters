#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 main._.py
echo "Results written under results/"
