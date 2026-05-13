#!/bin/bash
# GPU Training Launch Script — NVIDIA L40S (48GB VRAM)
#
# Architecture:
#   Docker:  MLflow server (port 8002)
#   tmux window 1: Traditional ML (CPU, joblib parallel)     ~30-60 min
#   tmux window 2: DL + Transformers (GPU L40S, sequential)  ~10-12 hours
#
# Checkpoint: safe to re-run if interrupted (skips completed runs)
# Reset:     rm -rf checkpoints/

DATA_PATH="${1:-./data/new_all.json}"

VENV_PATH="${2:-./venv}"

mkdir -p checkpoints logs mlflow_data

echo "Starting MLflow server (Docker)..."
docker compose up -d mlflow-server
sleep 3

export MLFLOW_TRACKING_URI=http://localhost:8002

echo "Waiting for MLflow to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8002/health > /dev/null 2>&1 || curl -s http://localhost:8002 > /dev/null 2>&1; then
        echo "MLflow ready at http://localhost:8002"
        break
    fi
    sleep 2
done

tmux new-session -d -s train -n traditional

# Window 1: Traditional ML (CPU only, joblib parallel)
tmux send-keys -t train:traditional "source $VENV_PATH/bin/activate && export MLFLOW_TRACKING_URI=http://localhost:8002 && python run_pipeline.py --data_path $DATA_PATH --run traditional 2>&1 | tee logs/traditional.log" Enter

# Window 2: DL then Transformers (GPU sequential)
tmux new-window -t train -n gpu
tmux send-keys -t train:gpu "source $VENV_PATH/bin/activate && export MLFLOW_TRACKING_URI=http://localhost:8002 && echo '=== Deep Learning (8 runs) ===' && python run_pipeline.py --data_path $DATA_PATH --run dl 2>&1 | tee logs/dl.log && echo '=== Transformers (80 runs) ===' && python run_pipeline.py --data_path $DATA_PATH --run transformers 2>&1 | tee logs/transformers.log && echo '=== All GPU training complete ==='" Enter

# Window 3: Monitor
tmux new-window -t train -n monitor
tmux send-keys -t train:monitor "source $VENV_PATH/bin/activate && watch -n 10 'echo === GPU === && nvidia-smi && echo && echo === Checkpoints === && for f in checkpoints/*.json; do echo \$f: \$(python -c \"import json; d=json.load(open(\\\"\$f\\\")); print(len(d.get(\\\"completed_runs\\\",[])),\\\"completed\\\")\" 2>/dev/null); done'" Enter

echo ""
echo "All launched in tmux session 'train':"
echo "  Window 'traditional' → CPU (joblib n_jobs=4)"
echo "  Window 'gpu'         → GPU L40S (DL → Transformers, sequential)"
echo "  Window 'monitor'      → GPU + checkpoint status"
echo ""
echo "MLflow UI: http://localhost:8002"
echo ""
echo "Commands:"
echo "  tmux attach -t train              # attach to see all"
echo "  Ctrl+B then D                     # detach (training keeps running)"
echo "  tmux attach -t train:gpu           # watch GPU progress"
echo "  tmux attach -t train:monitor        # watch GPU usage + progress"
echo ""
echo "If disconnected: re-run this script — checkpoint skips completed runs"
echo "Stop MLflow:     docker compose down"
