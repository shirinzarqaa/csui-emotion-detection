#!/bin/bash
# Save experiment results to a dated folder for GitHub
# Usage: bash save_experiment.sh exp_1

EXP_NAME="${1:-exp_1}"

echo "Saving experiment results to ${EXP_NAME}/..."

mkdir -p "${EXP_NAME}/analysis"
mkdir -p "${EXP_NAME}/checkpoints"
mkdir -p "${EXP_NAME}/logs"
mkdir -p "${EXP_NAME}/mlflow"

# Copy analysis files (CSV + XLSX)
cp -v analysis/*.csv "${EXP_NAME}/analysis/" 2>/dev/null
cp -v analysis/*.xlsx "${EXP_NAME}/analysis/" 2>/dev/null

# Copy checkpoint files
cp -v checkpoints/*.json "${EXP_NAME}/checkpoints/" 2>/dev/null

# Copy log files
cp -v logs/*.log "${EXP_NAME}/logs/" 2>/dev/null

# Export MLflow database
if [ -f "mlflow_data/mlflow.db" ]; then
    cp -v mlflow_data/mlflow.db "${EXP_NAME}/mlflow/" 2>/dev/null
fi

# Generate summary
source venv/bin/activate 2>/dev/null
python3 -c "
import json, os, glob

exp_dir = '${EXP_NAME}'

# Checkpoint summary
for ckpt_file in glob.glob(os.path.join(exp_dir, 'checkpoints', '*.json')):
    name = os.path.basename(ckpt_file).replace('_checkpoint.json', '')
    with open(ckpt_file) as f:
        d = json.load(f)
    runs = d.get('completed_runs', [])
    p1 = d.get('phase_1_complete', False)
    p2 = d.get('phase_2_complete', False)
    results = d.get('phase1_results', [])
    best_f1 = max((r.get('val_f1_macro', 0) for r in results), default=0)
    print(f'{name}: {len(runs)} runs | Phase1={p1} Phase2={p2} | Best val_f1_macro={best_f1:.4f}')

# Analysis file count
csv_count = len(glob.glob(os.path.join(exp_dir, 'analysis', '*.csv')))
xlsx_count = len(glob.glob(os.path.join(exp_dir, 'analysis', '*.xlsx')))
print(f'\nAnalysis: {csv_count} CSV, {xlsx_count} XLSX')

# MLflow summary
try:
    import mlflow
    mlflow.set_tracking_uri('http://localhost:8002')
    print('\nMLflow Experiments:')
    for exp in mlflow.search_experiments():
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        if len(runs) > 0:
            f1_cols = [c for c in runs.columns if 'f1_macro' in c]
            if f1_cols:
                best = runs.sort_values(by=f1_cols[0], ascending=False).iloc[0]
                run_name = best.get('tags.mlflow.runName', '?')
                best_f1 = best[f1_cols[0]]
                print(f'  {exp.name}: {len(runs)} runs | Best: {run_name} (F1={best_f1:.4f})')
            else:
                print(f'  {exp.name}: {len(runs)} runs')
except Exception as e:
    print(f'MLflow summary failed: {e}')
"

echo ""
echo "Done! Results saved to ${EXP_NAME}/"
echo "To push: git add ${EXP_NAME}/ && git commit -m 'results: experiment 1' && git push"
