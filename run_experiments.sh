#!/usr/bin/env bash
# =============================================================================
# SEGAb-YOLO Experiment Runner - Simplified
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-100}"
DATA="${DATA:-coco128.yaml}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-0}"
MODEL="${MODEL:-}"
DATASET="${DATASET:-}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
NAME="${NAME:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Parse command line arguments
COMMAND="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --data) DATA="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        --dry-run) DRY_RUN="true"; shift ;;
        --verbose) VERBOSE="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-100}"
DATA="${DATA:-coco128.yaml}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-0}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"
NAME="${NAME:-}"

echo "============================================================================="
echo "SEGAb-YOLO Experiment Runner"
echo "Project: $PROJECT_ROOT"
echo "Command: $COMMAND | Data: $DATA | Epochs: $EPOCHS | Device: $DEVICE"
echo "Model: ${MODEL:-all} | Dataset: ${DATASET:-all} | Limit: ${LIMIT:-none}"
echo "Batch: $BATCH | Img size: $IMGSZ | Workers: $WORKERS"
[[ -n "$NAME" ]] && echo "Run name: $NAME"
echo "============================================================================="

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
resolve() {
    local p="$1"
    if [[ ! "$p" = /* ]]; then
        echo "$PROJECT_ROOT/$p"
    else
        echo "$p"
    fi
}

run_train() {
    local model_yaml="$1"
    local name="${2:-$(basename "$model_yaml" .yaml)}"
    local epochs="${3:-$EPOCHS}"
    local data="${4:-$DATA}"
    local run_name="${5:-}"

    if [[ -n "$NAME" ]]; then
        run_name="$NAME"
    fi

    echo ""
    echo ">>> Training: ${name:-$model_yaml} ($epochs epochs on $data)"
    echo "-----------------------------------------------------------------------------"

    local model_path="$(resolve "segab_yolo/cfg/models/11/${model_yaml}")"
    local project_dir="runs/$(basename "$data" .yaml)"
    local run_name_final="${NAME:-$(basename "$model_yaml" .yaml)}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would train: $model_yaml on $data for $epochs epochs"
        return 0
    fi

    local python_cmd="${PYTHON:-python3}"
    $python_cmd -c "
from segab_yolo import YOLO
model = YOLO('$model_yaml')
model.train(
    data='$data',
    epochs=$epochs,
    batch=$BATCH,
    imgsz=$IMGSZ,
    device=$DEVICE,
    workers=$WORKERS,
    verbose=False,
    name='$run_name_final',
    project='runs',
    exist_ok=True
)
print('DONE: $run_name_final')
"
}

run_pipeline() {
    local dataset="$1"
    local model="${2:-}"
    local limit="${3:-}"
    local dry_run="${4:-false}"

    local python_cmd="${PYTHON:-python3}"
    local cmd="$python_cmd scripts/run_pipeline.py --dataset $dataset"
    [[ -n "$model" ]] && cmd="$cmd --model $model"
    [[ -n "$limit" ]] && cmd="$cmd --limit $limit"
    [[ "$dry_run" == "true" ]] && cmd="$cmd --dry-run"

    echo ""
    echo ">>> Pipeline: dataset=$dataset model=${model:-all} limit=${limit:-none}"
    echo "-----------------------------------------------------------------------------"

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY RUN] $cmd"
        return 0
    fi

    eval "$cmd"
}

run_validate() {
    local dataset="$1"
    local model="$2"

    local python_cmd="${PYTHON:-python3}"
    local cmd="$python_cmd scripts/validate_models.py --dataset $dataset"
    [[ -n "$model" ]] && cmd="$cmd --model $model"

    echo ""
    echo ">>> Validation: dataset=$dataset model=${model:-all}"
    eval "$cmd"
}

# Read model list from train_config.yaml
get_models() {
    local config="train_config.yaml"
    [[ -f "$config" ]] || { echo "train_config.yaml not found"; exit 1; }
    python3 -c "
import yaml, sys
with open('train_config.yaml') as f:
    cfg = yaml.safe_load(f)
for m in cfg.get('models', []):
    if isinstance(m, str):
        print(m)
    elif isinstance(m, dict) and 'name' in m:
        print(m['name'])
    else:
        print(m)
"
}

# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
cmd_train() {
    local data_override=""
    local epochs_override=""
    local model_filter=""
    local dataset_filter=""
    local name_override=""

    # Parse train-specific args
    while [[ $# -gt 0 ]]; do
        case $1 in
            --data) DATA="$2"; shift 2 ;;
            --epochs) EPOCHS="$2"; shift 2 ;;
            --model) MODEL="$2"; shift 2 ;;
            --dataset) DATASET="$2"; shift 2 ;;
            --name) NAME="$2"; shift 2 ;;
            *) break ;;
        esac
    done

    local data_override=""
    [[ -n "$DATASET" ]] && data_override="$DATA" || data_override=""

    if [[ -n "$MODEL" ]]; then
        # Train single model
        local model_yaml="${MODEL}.yaml"
        [[ "$MODEL" == *.yaml ]] && model_yaml="$MODEL"
        run_train "$model_yaml" "$MODEL" "$EPOCHS" "$DATA" "${NAME:-}"
    else
        # Train all models from config
        local models=($(get_models))
        for m in "${models[@]}"; do
            local yaml="${m}.yaml"
            [[ "$m" == *.yaml ]] && yaml="$m"
            run_train "$yaml" "$m" "$EPOCHS" "$DATA" "${NAME:-}"
        done
    fi
}

cmd_pipeline() {
    [[ -z "$DATASET" ]] && { echo "Error: --dataset required for pipeline"; exit 1; }

    local dataset="$DATASET"
    local model="${MODEL:-}"
    local limit="${LIMIT:-}"
    local dry_run="${DRY_RUN:-false}"

    run_pipeline "$dataset" "$model" "$limit" "$DRY_RUN"
}

cmd_validate() {
    [[ -z "$DATASET" || -z "$MODEL" ]] && { echo "Error: --dataset and --model required for validate"; exit 1; }
    run_validate "$DATASET" "$MODEL"
}

cmd_pipeline_full() {
    local data_override="${DATA:-}"
    [[ -n "$DATASET" ]] && data_override="$DATASET"

    local models=(
        "yolo11n"
        "yolo11_gam_coordatt_full"
        "yolo11_simam_triplet_full"
        "yolo11_lsk_coordatt_full"
        "yolo11_coordatt_min"
        "yolo11_triplet_min"
        "yolo11_gam_min"
        "yolo26n"
    )

    local dataset="${data_override:-coco128.yaml}"

    echo "============================================================================="
    echo "FULL PIPELINE: $EPOCHS epochs on $dataset"
    echo "============================================================================="

    for model in "${models[@]}"; do
        run_pipeline "$dataset" "$model" "" "false"
    done
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  train              Train models
        --data PATH        Dataset YAML (default: coco128.yaml)
        --epochs N         Epochs (default: 100)
        --model NAME       Single model to train (default: all from config)
        --dataset NAME     Dataset filter from config (optional)
        --name NAME        Custom run name (default: model name)

  pipeline           Run inference → XAI → viz pipeline
        --dataset PATH     Dataset YAML path (required)
        --model NAME       Model to run (default: all from config)
        --limit N          Limit images (default: all)
        --dry-run          Print commands without executing

  validate           Quick validation (1 epoch, 50 images)
        --dataset PATH     Dataset YAML path (required)
        --model NAME       Model to validate (required)

  pipeline-full      Run full pipeline on all benchmark models
        --data PATH        Dataset YAML (default: DATA env)
        --epochs N         Epochs for training step (default: 100)

Global options:
  --data PATH          Dataset YAML (default: coco128.yaml)
  --epochs N           Epochs (default: 100)
  --model NAME         Model name filter
  --dataset NAME       Dataset name filter
  --limit N            Limit images for infer/XAI
  --device N           GPU device (default: 0)
  --batch N            Batch size (default: 16)
  --imgsz N            Image size (default: 640)
  --workers N          DataLoader workers (default: 0)
  --name NAME          Custom run name
  --dry-run            Print commands without executing
  --verbose            Verbose output

Environment:
  DEVICE=0         GPU device
  EPOCHS=100       Default epochs
  DATA=coco128.yaml Default dataset
  BATCH=16         Batch size
  IMGSZ=640        Image size
  WORKERS=0        Workers
  PYTHON=python    Python executable

Examples:
  $0 train --data data/my.yaml --epochs 100 --model yolo11n
  $0 train --data data/my.yaml --epochs 100
  $0 pipeline --data data/my.yaml --model yolo11n --limit 100
  $0 validate --data data/my.yaml --model yolo11n
  $0 pipeline-full --data data/my.yaml --epochs 50
EOF
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
case "${COMMAND:-}" in
    train)
        cmd_train "$@"
        ;;
    pipeline)
        cmd_pipeline
        ;;
    validate)
        cmd_validate
        ;;
    pipeline-full)
        cmd_pipeline_full
        ;;
    *)
        usage
        exit 1
        ;;
esac

echo ""
echo "============================================================================="
echo "Done."
echo "============================================================================="