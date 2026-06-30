#!/usr/bin/env bash
# Download PETA benchmark_datasets.zip into ft_datasets/
#
# Source: https://github.com/mingchen-li/ProteinPretraining
# Google Drive link from PETA README:
#   https://drive.google.com/file/d/1o1yIE18WPOVJ8gBL5xcZEldtLSazRVYb/view

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="${PETA_DATA_DIR:-$PWD/ft_datasets}"
ZIP="${1:-benchmark_datasets.zip}"

echo "PETA data destination: $DEST"

if [[ -d "$DEST/flip" && -d "$DEST/tape" ]]; then
  echo "ft_datasets already present — nothing to do."
  exit 0
fi

if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  echo ""
  echo "Download benchmark_datasets.zip from:"
  echo "  https://drive.google.com/file/d/1o1yIE18WPOVJ8gBL5xcZEldtLSazRVYb/view"
  echo ""
  echo "Then run:"
  echo "  unzip benchmark_datasets.zip -d ."
  echo "  # should create ./ft_datasets/"
  echo ""
  echo "Or pass the zip path:"
  echo "  ./scripts/setup_peta_data.sh /path/to/benchmark_datasets.zip"
  exit 1
fi

unzip -q "$ZIP" -d "$(dirname "$DEST")"
echo "Done. Set: export PETA_DATA_DIR=$DEST"
echo "List tasks: python run.py list-tasks"
