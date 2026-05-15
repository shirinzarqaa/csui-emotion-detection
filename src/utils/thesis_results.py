import os
import pandas as pd
import mlflow
from loguru import logger


def generate_thesis_tables(tracking_uri="http://localhost:8002", output_path="thesis_results"):
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
            clean_exp_name = exp_name.replace("_exp2", "")

            if "Traditional" in exp_name:
                pipeline = "Traditional ML"
            elif "Deep_Learning" in exp_name:
                pipeline = "Deep Learning"
            elif "Transformer" in exp_name:
                pipeline = "Transformers"
            else:
                pipeline = "Other"

            is_test = "Final_Test" in exp_name

            row = {
                "experiment": exp_tag,
                "pipeline": pipeline,
                "phase": "Phase 2 (Test)" if is_test else "Phase 1 (Val)",
                "run_name": run.get("tags.mlflow.runName", ""),
            }

            metric_keys = [
                ("F1-Macro", ["val_f1_macro", "test_f1_macro"]),
                ("F1-Micro", ["val_f1_micro", "test_f1_micro"]),
                ("F1-Weighted", ["val_f1_weighted", "test_f1_weighted"]),
                ("Hamming Loss", ["val_hamming_loss", "test_hamming_loss"]),
                ("Exact Match Ratio", ["val_subset_accuracy", "test_subset_accuracy"]),
            ]

            for nice_name, keys in metric_keys:
                val_key = f"metrics.{keys[0]}"
                test_key = f"metrics.{keys[1]}"

                if is_test:
                    val = run.get(test_key, None)
                else:
                    val = run.get(val_key, None)

                row[nice_name] = round(val, 4) if val is not None else None

            param_cols = [c for c in runs.columns if c.startswith("params.")]
            for pc in param_cols:
                nice = pc.replace("params.", "")
                if run.get(pc) is not None:
                    row[f"param_{nice}"] = run[pc]

            all_rows.append(row)

    if not all_rows:
        logger.error("No runs found in MLflow!")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(output_path, exist_ok=True)

    metric_cols = ["F1-Macro", "F1-Micro", "F1-Weighted", "Hamming Loss", "Exact Match Ratio"]

    for exp_tag in ["exp1", "exp2"]:
        exp_df = df[df["experiment"] == exp_tag].copy()

        if exp_df.empty:
            logger.warning(f"No runs for {exp_tag}")
            continue

        display_cols = ["pipeline", "phase", "run_name"] + metric_cols
        param_cols_found = [c for c in exp_df.columns if c.startswith("param_") and exp_df[c].notna().any()]
        display_cols += param_cols_found
        available = [c for c in display_cols if c in exp_df.columns]

        exp_df = exp_df[available].sort_values(["pipeline", "phase", "F1-Macro"], ascending=[True, True, False])

        exp_df.to_csv(f"{output_path}/{exp_tag}_all_results.csv", index=False)
        exp_df.to_excel(f"{output_path}/{exp_tag}_all_results.xlsx", index=False, engine='openpyxl')
        logger.info(f"{exp_tag}: {len(exp_df)} runs saved")

    with pd.ExcelWriter(f"{output_path}/thesis_results.xlsx", engine='openpyxl') as writer:
        for exp_tag in ["exp1", "exp2"]:
            csv_path = f"{output_path}/{exp_tag}_all_results.csv"
            if os.path.exists(csv_path):
                pd.read_csv(csv_path).to_excel(writer, sheet_name=exp_tag, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Thesis tables saved to: {output_path}/")
    logger.info(f"  - exp1_all_results.csv/.xlsx (all Phase 1 + Phase 2 runs)")
    logger.info(f"  - exp2_all_results.csv/.xlsx (all Phase 1 + Phase 2 runs)")
    logger.info(f"  - thesis_results.xlsx (both in one file)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:8002")
    parser.add_argument("--output", type=str, default="thesis_results")
    args = parser.parse_args()

    generate_thesis_tables(tracking_uri=args.mlflow_uri, output_path=args.output)
