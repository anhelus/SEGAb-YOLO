#!/usr/bin/env bash
# =============================================================================
# SEGAb-YOLO Experiment Runner
# Runs all attention mechanism experiments on tomatoes & lettuces datasets
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-0}"
EPOCHS_FULL="${EPOCHS_FULL:-100}"
EPOCHS_QUICK="${EPOCHS_QUICK:-3}"
BATCH="${BATCH:-16}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-0}"

echo "============================================================================="
echo "SEGAb-YOLO Experiment Runner"
echo "Project: $PROJECT_ROOT"
echo "Device: $DEVICE | Epochs full: $EPOCHS_FULL | Quick: $EPOCHS_QUICK"
echo "============================================================================="

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
run_train() {
    local model_yaml="$1"
    local name="${2:-$(basename "$model_yaml" .yaml)}"
    local epochs="${3:-$EPOCHS_QUICK}"
    local data="${4:-coco128.yaml}"
    
    echo ""
    echo ">>> Training: $name ($epochs epochs on $data)"
    echo "-----------------------------------------------------------------------------"
    $PYTHON -c "
from segab_yolo import YOLO
model = YOLO('segab_yolo/cfg/models/11/$model_yaml')
model.train(data='$data', epochs=$epochs, batch=$BATCH, imgsz=$IMGSZ, device=$DEVICE, workers=$WORKERS, verbose=False, name='$name')
print('DONE: $name')
"
}

run_pipeline() {
    local dataset="$1"
    local model="${2:-}"
    local limit="${3:-}"
    local dry_run="${4:-false}"
    
    local cmd="$PYTHON scripts/run_pipeline.py --dataset $dataset"
    [[ -n "$model" ]] && cmd="$cmd --model $model"
    [[ -n "$limit" ]] && cmd="$cmd --limit $limit"
    [[ "$dry_run" == "true" ]] && cmd="$cmd --dry-run"
    
    echo ""
    echo ">>> Pipeline: dataset=$dataset model=${model:-all} limit=${limit:-none} dry_run=$dry_run"
    echo "-----------------------------------------------------------------------------"
    eval "$cmd"
}

# -----------------------------------------------------------------------------
# PHASE 1: Singoli attention (neck full + backbone) - quick test on coco128
# -----------------------------------------------------------------------------
phase1_quick() {
    echo "============================================================================="
    echo "PHASE 1: Quick test (3 epochs on coco128) - Singoli attention"
    echo "============================================================================="
    
    local models=(
        "yolo11n.yaml yolo11n_baseline"
        "yolo11_gam_n.yaml yolo11_gam_n"
        "yolo11_gam_bbone_n.yaml yolo11_gam_bbone_n"
        "yolo11_simam_n.yaml yolo11_simam_n"
        "yolo11_coordatt_min.yaml yolo11_coordatt_min"
        "yolo11_triplet_min.yaml yolo11_triplet_min"
        "yolo11_lsk_coordatt_full.yaml yolo11_lsk_coordatt_full"
    )
    
    for entry in "${models[@]}"; do
        read -r yaml name <<< "$entry"
        run_train "$yaml" "$name" "$EPOCHS_QUICK" "coco128.yaml"
    done
}

# -----------------------------------------------------------------------------
# PHASE 2: Min variants (solo P3) - quick test
# -----------------------------------------------------------------------------
phase2_quick() {
    echo "============================================================================="
    echo "PHASE 2: Quick test (3 epochs on coco128) - Min variants (P3 only)"
    echo "============================================================================="
    
    local models=(
        "yolo11_gam_min.yaml yolo11_gam_min"
        "yolo11_coordatt_min.yaml yolo11_coordatt_min"
        "yolo11_triplet_min.yaml yolo11_triplet_min"
    )
    
    for entry in "${models[@]}"; do
        read -r yaml name <<< "$entry"
        run_train "$yaml" "$name" "$EPOCHS_QUICK" "coco128.yaml"
    done
}

# -----------------------------------------------------------------------------
# PHASE 3: Combo backbone+neck - quick test
# -----------------------------------------------------------------------------
phase3_quick() {
    echo "============================================================================="
    echo "PHASE 3: Quick test (3 epochs on coco128) - Combo backbone+neck"
    echo "============================================================================="
    
    local models=(
        "yolo11_gam_coordatt_full.yaml yolo11_gam_coordatt_full"
        "yolo11_simam_triplet_full.yaml yolo11_simam_triplet_full"
        "yolo11_lsk_coordatt_full.yaml yolo11_lsk_coordatt_full"
    )
    
    for entry in "${models[@]}"; do
        read -r yaml name <<< "$entry"
        run_train "$yaml" "$name" "$EPOCHS_QUICK" "coco128.yaml"
    done
}

# -----------------------------------------------------------------------------
# PHASE 4: Full training on tomatoes & lettuces (100 epochs)
# -----------------------------------------------------------------------------
phase4_full() {
    local dataset="$1"
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
    
    echo "============================================================================="
    echo "PHASE 4: Full training ($EPOCHS_FULL epochs) on $dataset"
    echo "============================================================================="
    
    for model in "${models[@]}"; do
        run_pipeline "$dataset" "$model" "" "false"
    done
}

# -----------------------------------------------------------------------------
# Quick validation run (limit images, dry-run or few epochs)
# -----------------------------------------------------------------------------
validate_quick() {
    local dataset="$1"
    local model="$2"
    
    echo "============================================================================="
    echo "QUICK VALIDATION: $dataset / $model (limit 50, 1 epoch)"
    echo "============================================================================="
    
    run_pipeline "$dataset" "$model" "50" "false"
}

# -----------------------------------------------------------------------------
# Main menu / dispatch
# -----------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  phase1          Quick test Phase 1 (7 singoli, 3 epoche su coco128)
  phase2          Quick test Phase 2 (3 min variants, 3 epoche su coco128)
  phase3          Quick test Phase 3 (3 combo, 3 epoche su coco128)
  quick-all       Run phase1 + phase2 + phase3 sequentially
  
  full-tomatoes   Full training on tomatoes (100 epoche, all models)
  full-lettuces   Full training on lettuces (100 epoche, all models)
  full-both       Full training on both datasets
  
  validate        Quick validation on dataset (limit 50 imgs)
                    Usage: $0 validate <tomatoes|lettuces> <model_name>
  
  pipeline        Run full pipeline (infer → xai → viz) on dataset
                    Usage: $0 pipeline <tomatoes|lettuces> [model] [limit]
  
  dry-run         Dry-run pipeline to check config
                    Usage: $0 dry-run <tomatoes|lettuces> [model]

Environment variables:
  DEVICE=0              GPU device (default: 0)
  EPOCHS_FULL=100       Full training epochs
  EPOCHS_QUICK=3        Quick test epochs
  BATCH=16              Batch size
  IMGSZ=640             Image size
  WORKERS=0             DataLoader workers
  PYTHON=python         Python executable

Examples:
  $0 phase1
  $0 quick-all
  $0 full-tomatoes
  $0 validate tomatoes yolo11_gam_coordatt_full
  $0 pipeline tomatoes yolo11_gam_coordatt_full 100
  $0 dry-run tomatoes
  DEVICE=1 EPOCHS_FULL=50 $0 full-lettuces
EOF
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
case "${1:-}" in
    phase1)
        phase1_quick
        ;;
    phase2)
        phase2_quick
        ;;
    phase3)
        phase3_quick
        ;;
    quick-all)
        phase1_quick
        phase2_quick
        phase3_quick
        ;;
    full-tomatoes)
        phase4_full "tomatoes"
        ;;
    full-lettuces)
        phase4_full "lettuces"
        ;;
    full-both)
        phase4_full "tomatoes"
        phase4_full "lettuces"
        ;;
    validate)
        [[ -z "${2:-}" || -z "${3:-}" ]] && { usage; exit 1; }
        validate_quick "$2" "$3"
        ;;
    pipeline)
        [[ -z "${2:-}" ]] && { usage; exit 1; }
        run_pipeline "$2" "${3:-}" "${4:-}" "false"
        ;;
    dry-run)
        [[ -z "${2:-}" ]] && { usage; exit 1; }
        run_pipeline "$2" "${3:-}" "" "true"
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