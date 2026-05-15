import os
import pandas as pd
import mlflow
from loguru import logger


def generate_thesis_tables(
    tracking_uri="http://localhost:8002",
    output_path="thesis_results",
    eval_dir=None,
):
    mlflow.set_tracking_uri(tracking_uri)

    experiments = {}
    for exp in mlflow.search_experiments():
        experiments[exp.name] = exp.experiment_id

    all_rows = []
    for exp_name, exp_id in experiments.items():
        runs = mlflow.search_runs(experiment_ids=[exp_id])
        if runs.empty:
            continue

        for _, run in runs.iterrows():
            is_exp2 = "_exp2" in exp_name
            exp_tag = "exp2" if is_exp2 else "exp1"

            if "Traditional" in exp_name:
                pipeline = "Traditional ML"
            elif "Deep_Learning" in exp_name:
                pipeline = "Deep Learning"
            elif "Transformer" in exp_name:
                pipeline = "Transformers"
            else:
                pipeline = "Other"

            is_test = "Final_Test" in exp_name
            phase = "Phase 2 (Test)" if is_test else "Phase 1 (Val)"

            row = {
                "experiment": exp_tag,
                "pipeline": pipeline,
                "phase": phase,
                "run_name": run.get("tags.mlflow.runName", ""),
            }

            metric_map = {
                "Val F1-Macro": "metrics.val_f1_macro",
                "Val F1-Micro": "metrics.val_f1_micro",
                "Val F1-Weighted": "metrics.val_f1_weighted",
                "Val Hamming Loss": "metrics.val_hamming_loss",
                "Val Exact Match Ratio": "metrics.val_subset_accuracy",
                "Test F1-Macro": "metrics.test_f1_macro",
                "Test F1-Micro": "metrics.test_f1_micro",
                "Test F1-Weighted": "metrics.test_f1_weighted",
                "Test Hamming Loss": "metrics.test_hamming_loss",
                "Test Exact Match Ratio": "metrics.test_subset_accuracy",
            }

            for nice_name, mlflow_key in metric_map.items():
                val = run.get(mlflow_key, None)
                row[nice_name] = round(val, 4) if val is not None else None

            param_cols = [c for c in runs.columns if c.startswith("params.")]
            for pc in param_cols:
                nice = pc.replace("params.", "")
                if run.get(pc) is not None:
                    row[f"param_{nice}"] = run[pc]

            all_rows.append(row)

    eval_dfs = {}
    if eval_dir and os.path.exists(eval_dir):
        for exp_tag in ["exp1", "exp2"]:
            csv_path = f"{eval_dir}/{exp_tag}_all_val_test.csv"
            if os.path.exists(csv_path):
                eval_dfs[exp_tag] = pd.read_csv(csv_path)
                logger.info(f"Loaded evaluate_test results for {exp_tag}: {len(eval_dfs[exp_tag])} runs")

    if not all_rows and not eval_dfs:
        logger.error("No runs found in MLflow or eval_dir!")
        return

    mlflow_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    os.makedirs(output_path, exist_ok=True)

    val_metric_cols = ["Val F1-Macro", "Val F1-Micro", "Val F1-Weighted", "Val Hamming Loss", "Val Exact Match Ratio"]
    test_metric_cols = ["Test F1-Macro", "Test F1-Micro", "Test F1-Weighted", "Test Hamming Loss", "Test Exact Match Ratio"]

    for exp_tag in ["exp1", "exp2"]:
        combined_rows = []

        if not mlflow_df.empty:
            exp_mlflow = mlflow_df[mlflow_df["experiment"] == exp_tag].copy()
            if not exp_mlflow.empty:
                display_cols_mlflow = ["pipeline", "phase", "run_name"] + val_metric_cols + test_metric_cols
                param_cols_found = [c for c in exp_mlflow.columns if c.startswith("param_") and exp_mlflow[c].notna().any()]
                display_cols_mlflow += param_cols_found
                available = [c for c in display_cols_mlflow if c in exp_mlflow.columns]

                exp_mlflow = exp_mlflow[available].sort_values(
                    ["pipeline", "phase", "Val F1-Macro", "Test F1-Macro"],
                    ascending=[True, True, False, False]
                )
                combined_rows.append(exp_mlflow)

        if exp_tag in eval_dfs:
            combined_rows.append(eval_dfs[exp_tag])

        if combined_rows:
            exp_df = pd.concat(combined_rows, ignore_index=True).drop_duplicates(subset=["run_name"], keep="last")
        else:
            continue

        exp_df.to_csv(f"{output_path}/{exp_tag}_all_results.csv", index=False)
        exp_df.to_excel(f"{output_path}/{exp_tag}_all_results.xlsx", index=False, engine='openpyxl')
        logger.info(f"{exp_tag}: {len(exp_df)} runs saved")

    summary_rows = []
    for exp_tag in ["exp1", "exp2"]:
        for pipeline in ["Traditional ML", "Deep Learning", "Transformers"]:
            subset = mlflow_df[(mlflow_df["experiment"] == exp_tag) & (mlflow_df["pipeline"] == pipeline)] if not mlflow_df.empty else pd.DataFrame()
            if exp_tag in eval_dfs:
                eval_subset = eval_dfs[exp_tag][eval_dfs[exp_tag]["pipeline"] == pipeline] if "pipeline" in eval_dfs[exp_tag].columns else pd.DataFrame()
            else:
                eval_subset = pd.DataFrame()

            p1 = subset[subset["phase"] == "Phase 1 (Val)"] if not subset.empty else pd.DataFrame()
            p2 = subset[subset["phase"] == "Phase 2 (Test)"] if not subset.empty else pd.DataFrame()

            best_val_f1 = p1["Val F1-Macro"].max() if not p1.empty and "Val F1-Macro" in p1.columns else None
            mean_val_f1 = p1["Val F1-Macro"].mean() if not p1.empty and "Val F1-Macro" in p1.columns else None

            best_test_f1 = None
            mean_test_f1 = None
            if not p2.empty and "Test F1-Macro" in p2.columns:
                best_test_f1 = p2["Test F1-Macro"].max()
                mean_test_f1 = p2["Test F1-Macro"].mean()
            elif not eval_subset.empty and "Test f1_macro" in eval_subset.columns:
                best_test_f1 = eval_subset["Test f1_macro"].max()
                mean_test_f1 = eval_subset["Test f1_macro"].mean()

            num_val = len(p1) if not p1.empty else len(eval_subset) if not eval_subset.empty else 0
            num_test = len(p2) if not p2.empty else len(eval_subset) if not eval_subset.empty else 0

            summary_rows.append({
                "experiment": exp_tag,
                "pipeline": pipeline,
                "num_val_runs": num_val,
                "num_test_runs": num_test,
                "best_val_f1_macro": round(best_val_f1, 4) if best_val_f1 is not None else None,
                "mean_val_f1_macro": round(mean_val_f1, 4) if mean_val_f1 is not None else None,
                "best_test_f1_macro": round(best_test_f1, 4) if best_test_f1 is not None else None,
                "mean_test_f1_macro": round(mean_test_f1, 4) if mean_test_f1 is not None else None,
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(f"{output_path}/summary.csv", index=False)
        summary_df.to_excel(f"{output_path}/summary.xlsx", index=False, engine='openpyxl')

    with pd.ExcelWriter(f"{output_path}/thesis_results.xlsx", engine='openpyxl') as writer:
        for exp_tag in ["exp1", "exp2"]:
            csv_path = f"{output_path}/{exp_tag}_all_results.csv"
            if os.path.exists(csv_path):
                pd.read_csv(csv_path).to_excel(writer, sheet_name=exp_tag, index=False)
        if summary_rows:
            summary_df.to_excel(writer, sheet_name="summary", index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Thesis tables saved to: {output_path}/")
    logger.info(f"  - exp1_all_results.csv/.xlsx")
    logger.info(f"  - exp2_all_results.csv/.xlsx")
    logger.info(f"  - summary.csv/.xlsx")
    logger.info(f"  - thesis_results.xlsx (all in one file)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:8002")
    parser.add_argument("--output", type=str, default="thesis_results")
    parser.add_argument("--eval_dir", type=str, default="thesis_results")
    args = parser.parse_args()

    generate_thesis_tables(
        tracking_uri=args.mlflow_uri,
        output_path=args.output,
        eval_dir=args.eval_dir,
    )
