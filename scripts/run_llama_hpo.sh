#!/bin/bash
# scripts/run_llama_hpo.sh
#
# Convenience wrapper for running UGRO HPO sweeps with MLflow tracking.
# Provides production-ready defaults for LLaMA LoRA fine-tuning.
#
# Usage:
#   ./scripts/run_llama_hpo.sh                    # Use defaults
#   STUDY_NAME=my-study N_TRIALS=50 ./scripts/run_llama_hpo.sh
#
# Environment Variables:
#   STUDY_NAME      - Unique study name (default: llama2-7b-lora-v1)
#   SEARCH_SPACE    - Path to search space YAML (default: config/llama2_lora.yaml)
#   N_TRIALS        - Number of trials (default: 100)
#   PARALLEL_JOBS   - Concurrent trials (default: 8)
#   ALGORITHM       - Search algorithm: tpe, asha (default: asha)
#   RAY_GPU_PER_TRIAL - GPU fraction per trial (default: 0.5)
#   START_MLFLOW    - Start MLflow UI in background (default: true)

set -euo pipefail

# Configuration with defaults
STUDY_NAME="${STUDY_NAME:-llama2-7b-lora-v1}"
SEARCH_SPACE="${SEARCH_SPACE:-config/llama2_lora.yaml}"
N_TRIALS="${N_TRIALS:-100}"
PARALLEL_JOBS="${PARALLEL_JOBS:-8}"
ALGORITHM="${ALGORITHM:-asha}"
RAY_GPU_PER_TRIAL="${RAY_GPU_PER_TRIAL:-0.5}"
# Default to local Ray instance (empty) unless specified
RAY_ADDRESS="${RAY_ADDRESS:-}"
STORAGE_BACKEND="${STORAGE_BACKEND:-sqlite:///studies/${STUDY_NAME}.db}"
TRACKING_URI="${TRACKING_URI:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
START_MLFLOW="${START_MLFLOW:-true}"

# Create output directories
mkdir -p studies results config

echo "============================================================"
echo "UGRO HPO: ${STUDY_NAME}"
echo "============================================================"
echo "Search Space: ${SEARCH_SPACE}"
echo "Trials: ${N_TRIALS} (${PARALLEL_JOBS} parallel)"
echo "Algorithm: ${ALGORITHM}"
echo "Storage: ${STORAGE_BACKEND}"
echo "============================================================"

# Check dependencies
check_deps() {
    if ! command -v pixi &> /dev/null; then
        echo "❌ Error: 'pixi' not found. Please install pixi first."
        exit 1
    fi
    
    # Check if HPO environment is solvable/available (optional but good)
    # pixi run -e hpo true &> /dev/null || echo "⚠️ Warning: 'hpo' environment might not be ready. Run 'pixi install' first."
}

check_deps

# MLflow UI (optional)
MLFLOW_PID=""
if [[ "${START_MLFLOW}" == "true" ]]; then
    if ! lsof -i :5000 &> /dev/null; then
        echo "Starting MLflow UI via pixi..."
        pixi run -e hpo mlflow ui --host 0.0.0.0 --port 5000 &
        MLFLOW_PID=$!
        TRACKING_URI="${TRACKING_URI:-http://localhost:5000}"
        echo "MLflow UI running at ${TRACKING_URI} (PID: ${MLFLOW_PID})"
        sleep 3  # Allow startup
    else
        echo "ℹ️  Port 5000 already in use. Assuming MLflow is running."
        # Don't set PID so we don't kill it
        TRACKING_URI="${TRACKING_URI:-http://localhost:5000}"
    fi
fi

# Cleanup trap
cleanup() {
    if [[ -n "${MLFLOW_PID}" ]]; then
        echo "Stopping MLflow UI..."
        kill "${MLFLOW_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Build command
# Using 'pixi run -e hpo' to ensure correct environment
CMD="pixi run -e hpo ugro hpo sweep"
CMD="${CMD} --study-name ${STUDY_NAME}"
CMD="${CMD} --search-space ${SEARCH_SPACE}"
CMD="${CMD} --n-trials ${N_TRIALS}"
CMD="${CMD} --parallel-jobs ${PARALLEL_JOBS}"
CMD="${CMD} --algorithm ${ALGORITHM}"
CMD="${CMD} --ray-gpu ${RAY_GPU_PER_TRIAL}"
CMD="${CMD} --storage ${STORAGE_BACKEND}"

if [[ -n "${RAY_ADDRESS}" ]]; then
    CMD="${CMD} --ray-address ${RAY_ADDRESS}"
fi

if [[ -n "${TRACKING_URI}" ]]; then
    CMD="${CMD} --tracking-uri ${TRACKING_URI}"
fi

if [[ -n "${WANDB_PROJECT}" ]]; then
    CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
fi

# Export paths
EXPORT_BEST="config/best_lora_${STUDY_NAME}.yaml"
TRIALS_CSV="results/${STUDY_NAME}_trials.csv"
CMD="${CMD} --export-best ${EXPORT_BEST}"
CMD="${CMD} --save-trials ${TRIALS_CSV}"

# Run sweep
echo ""
echo "Running: ${CMD}"
echo ""
# Execute via bash to handle variable expansion correctly if CMD contains spaces
eval "${CMD}"

# Analysis
echo ""
echo "============================================================"
echo "Analyzing results..."
echo "============================================================"
pixi run -e hpo python -c "
from ugro.hpo.analysis import analyze_hpo_results
analyze_hpo_results('${STORAGE_BACKEND}', '${STUDY_NAME}', 'results/')
"

echo ""
echo "============================================================"
echo "HPO Complete!"
echo "============================================================"
echo "Best config: ${EXPORT_BEST}"
echo "All trials: ${TRIALS_CSV}"
echo "Analysis: results/hpo_analysis_${STUDY_NAME}.png"
echo "============================================================"
