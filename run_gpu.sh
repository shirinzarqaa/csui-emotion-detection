#!/bin/bash
# GPU Training Launch Script — NVIDIA L40S (48GB VRAM)
#
# Architecture:
#   tmux window 1: Traditional ML (CPU, joblib parallel)     ~30-60 min
#   tmux window 2: DL + Transformers (GPU, sequential)       ~10-12 hours
#   Traditional runs on CPU in parallel with GPU tasks
#   DL and Transformers share the same GPU — must be sequential
#
# Checkpoint system: safe to re-run if interrupted (skips completed runs)
# Check progress:  cat checkpoints/*_checkpoint.json
# View metrics:    mlflow ui --host 0.0.0.0 --port 5000

DATA_PATH="${1:-./data/new_all.json}"
N_JOBS="${2:-4}"

mkdir -p checkpoints logs

tmux new-session -d -s train -n traditional

# Window 1: Traditional ML (CPU only, joblib parallel)
tmux send-keys -t train:traditional "python run_pipeline.py --data_path $DATA_PATH --run traditional 2>&1 | tee logs/traditional.log" Enter

# Window 2: DL then Transformers (GPU sequential — DL finishes first, then Transformers)
tmux new-window -t train -n gpu
tmux send-keys -t train:gpu "echo '=== Phase 1: Deep Learning (8 runs) ===' && python run_pipeline.py --data_path $DATA_PATH --run dl 2>&1 | tee logs/dl.log && echo '=== Phase 1: Transformers (80 runs) ===' && python run_pipeline.py --data_path $DATA_PATH --run transformers 2>&1 | tee logs/transformers.log && echo '=== All GPU training complete ==='" Enter

echo ""
echo "Launched in tmux session 'train':"
echo "  Window 'traditional' → CPU (joblib n_jobs=$N_JOBS)"
echo "  Window 'gpu'         → GPU L40S (DL → Transformers, sequential)"
echo ""
echo "Commands:"
echo "  tmux attach -t train              # attach to see all"
echo "  Ctrl+B then D                     # detach (training keeps running)"
echo "  tmux attach -t train:gpu           # watch GPU progress"
echo "  nvidia-smi -l 5                    # monitor GPU usage"
echo "  cat checkpoints/*_checkpoint.json  # check completed runs"
echo ""
echo "If disconnected: just re-run this script — checkpoint skips completed runs"
