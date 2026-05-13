#!/bin/bash
# Launch all 3 pipelines in parallel using tmux
# Usage: bash run_gpu.sh /path/to/new_all.json
# 
# Pipelines run in separate tmux panes — survive SSH disconnect.
# Check progress: tmux attach -t train
# Or check checkpoints: cat checkpoints/*_checkpoint.json

DATA_PATH="${1:-./data/new_all.json}"

mkdir -p checkpoints logs

tmux new-session -d -s train -n traditional

# Traditional ML
tmux send-keys -t train:traditional "python run_pipeline.py --data_path $DATA_PATH --run traditional 2>&1 | tee logs/traditional.log" Enter

# Deep Learning
tmux new-window -t train -n dl
tmux send-keys -t train:dl "python run_pipeline.py --data_path $DATA_PATH --run dl 2>&1 | tee logs/dl.log" Enter

# Transformers
tmux new-window -t train -n transformers
tmux send-keys -t train:transformers "python run_pipeline.py --data_path $DATA_PATH --run transformers 2>&1 | tee logs/transformers.log" Enter

echo "All 3 pipelines launched in tmux session 'train'"
echo ""
echo "Commands:"
echo "  tmux attach -t train              # attach to see logs"
echo "  Ctrl+B then D                     # detach (training keeps running)"
echo "  tmux attach -t train:traditional  # attach to specific pane"
echo "  cat checkpoints/*_checkpoint.json  # check progress"
echo "  mlflow ui --host 0.0.0.0 --port 5000  # view metrics"
