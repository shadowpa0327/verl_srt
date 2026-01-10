#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/plot_metrics_sweep.sh [--dir DIR] [--out-dir DIR] [--plot-script PATH] [--figsize W,H]
                                [--include-aggregate]

Examples:
  ./scripts/plot_metrics_sweep.sh \
    --dir /opt/tiger/verl_srt/results/metrics_sweep_20260111_003426

  ./scripts/plot_metrics_sweep.sh \
    --dir results/metrics_sweep_20260111_003426 \
    --out-dir results/metrics_sweep_20260111_003426/plots

Notes:
  - Produces one PNG per CSV (per-server).
  - Use --include-aggregate to additionally generate "_agg.png" aggregate plots.
  - Set env vars to override defaults: PLOT_SCRIPT, FIGSIZE
EOF
}

RESULTS_DIR=""
OUT_DIR=""
PLOT_SCRIPT="${PLOT_SCRIPT:-/opt/tiger/verl_srt/scripts/plot_metrics.py}"
FIGSIZE="${FIGSIZE:-14,10}"
DO_PER_SERVER=true
DO_AGG=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --plot-script)
      PLOT_SCRIPT="$2"
      shift 2
      ;;
    --figsize)
      FIGSIZE="$2"
      shift 2
      ;;
    --include-aggregate)
      DO_AGG=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="/opt/tiger/verl_srt/results/metrics_sweep_20260111_003426"
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="$RESULTS_DIR/plots"
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "ERROR: results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "ERROR: plot script not found: $PLOT_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

shopt -s nullglob
csvs=("$RESULTS_DIR"/metrics_*.csv)
shopt -u nullglob

if [[ ${#csvs[@]} -eq 0 ]]; then
  echo "ERROR: no metrics CSVs found under: $RESULTS_DIR (expected metrics_*.csv)" >&2
  exit 1
fi

echo "Found ${#csvs[@]} CSV(s) under $RESULTS_DIR"
echo "Output directory: $OUT_DIR"
echo "Plot script: $PLOT_SCRIPT"
echo "Figsize: $FIGSIZE"

for csv in "${csvs[@]}"; do
  base="$(basename "$csv" .csv)"

  if [[ "$DO_PER_SERVER" == "true" ]]; then
    out_png="$OUT_DIR/${base}.png"
    echo "Plot per-server: $csv -> $out_png"
    python "$PLOT_SCRIPT" "$csv" --output "$out_png" --figsize "$FIGSIZE"
  fi

  if [[ "$DO_AGG" == "true" ]]; then
    out_png_agg="$OUT_DIR/${base}_agg.png"
    echo "Plot aggregate: $csv -> $out_png_agg"
    python "$PLOT_SCRIPT" "$csv" --aggregate --output "$out_png_agg" --figsize "$FIGSIZE"
  fi
done

echo "Done."
