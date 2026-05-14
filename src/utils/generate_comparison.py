import os
import sys
import numpy as np
import pandas as pd
import mlflow
from loguru import logger
from sklearn.metrics import f1_score, precision_recall_fscore_support, hamming_loss, accuracy_score


def generate_comparison(tracking_uri="http://localhost:8002", output_path="experiment_comparison"):
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
            row = {
                "experiment_name": exp_name,
                "run_name": run.get("tags.mlflow.runName", ""),
                "phase": run.get("tags.phase", ""),
            }

            for col in run.columns:
                if col.startswith("params."):
                    row[col.replace("params.", "param_")] = run[col]
                elif col.startswith("metrics."):
                    row[col.replace("metrics.", "metric_")] = run[col]

            all_rows.append(row)

    if not all_rows:
        logger.error("No runs found in MLflow!")
        return

    df = pd.DataFrame(all_rows)

    def classify_pipeline(exp_name):
        if "Traditional" in exp_name:
            return "traditional"
        elif "Deep_Learning" in exp_name:
            return "dl"
        elif "Transformer" in exp_name:
            return "transformers"
        return "unknown"

    def classify_experiment(exp_name):
        if "_exp2" in exp_name:
            return "exp2"
        return "exp1"

    def classify_phase(exp_name):
        if "Final_Test" in exp_name:
            return "test"
        return "val"

    df["pipeline"] = df["experiment_name"].apply(classify_pipeline)
    df["experiment"] = df["experiment_name"].apply(classify_experiment)
    df["metric_phase"] = df["experiment_name"].apply(classify_phase)

    os.makedirs(output_path, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # SHEET 1 & 2: Val Summary (exp1, exp2)
    # ═══════════════════════════════════════════════════════════════
    val_cols = ["pipeline", "run_name", "experiment",
                "metric_val_f1_macro", "metric_val_f1_micro", "metric_val_f1_weighted",
                "metric_val_subset_accuracy", "metric_val_hamming_loss"]

    for exp_tag in ["exp1", "exp2"]:
        val_df = df[(df["experiment"] == exp_tag) & (df["metric_phase"] == "val")].copy()
        available_cols = [c for c in val_cols if c in val_df.columns]

        param_cols = [c for c in val_df.columns if c.startswith("param_")]
        available_cols += [c for c in param_cols if val_df[c].notna().any()]

        if not val_df.empty:
            sheet_df = val_df[available_cols].sort_values("metric_val_f1_macro", ascending=False)
            sheet_df.to_csv(f"{output_path}/{exp_tag}_val_summary.csv", index=False)
            sheet_df.to_excel(f"{output_path}/{exp_tag}_val_summary.xlsx", index=False, engine='openpyxl')
            logger.info(f"{exp_tag} val summary: {len(sheet_df)} runs saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 3 & 4: Test Summary (exp1, exp2)
    # ═══════════════════════════════════════════════════════════════
    test_cols = ["pipeline", "run_name", "experiment",
                 "metric_test_f1_macro", "metric_test_f1_micro", "metric_test_f1_weighted",
                 "metric_test_subset_accuracy", "metric_test_hamming_loss"]

    for exp_tag in ["exp1", "exp2"]:
        test_df = df[(df["experiment"] == exp_tag) & (df["metric_phase"] == "test")].copy()
        available_cols = [c for c in test_cols if c in test_df.columns]

        param_cols = [c for c in test_df.columns if c.startswith("param_")]
        available_cols += [c for c in param_cols if test_df[c].notna().any()]

        if not test_df.empty:
            sheet_df = test_df[available_cols].sort_values("metric_test_f1_macro", ascending=False)
            sheet_df.to_csv(f"{output_path}/{exp_tag}_test_summary.csv", index=False)
            sheet_df.to_excel(f"{output_path}/{exp_tag}_test_summary.xlsx", index=False, engine='openpyxl')
            logger.info(f"{exp_tag} test summary: {len(sheet_df)} runs saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 5: Comparison exp1 vs exp2
    # ═══════════════════════════════════════════════════════════════
    val_exp1 = df[(df["experiment"] == "exp1") & (df["metric_phase"] == "val")].copy()
    val_exp2 = df[(df["experiment"] == "exp2") & (df["metric_phase"] == "val")].copy()

    if not val_exp1.empty and not val_exp2.empty:
        comp_rows = []
        for pipeline in ["traditional", "dl", "transformers"]:
            for phase in ["val", "test"]:
                prefix = f"metric_{'val' if phase == 'val' else 'test'}_"
                e1 = df[(df["experiment"] == "exp1") & (df["metric_phase"] == phase) & (df["pipeline"] == pipeline)]
                e2 = df[(df["experiment"] == "exp2") & (df["metric_phase"] == phase) & (df["pipeline"] == pipeline)]

                if e1.empty and e2.empty:
                    continue

                for metric in ["f1_macro", "f1_micro", "f1_weighted", "subset_accuracy", "hamming_loss"]:
                    col = f"{prefix}{metric}"
                    e1_best = e1[col].max() if col in e1.columns and not e1.empty else None
                    e2_best = e2[col].max() if col in e2.columns and not e2.empty else None
                    e1_mean = e1[col].mean() if col in e1.columns and not e1.empty else None
                    e2_mean = e2[col].mean() if col in e2.columns and not e2.empty else None

                    comp_rows.append({
                        "pipeline": pipeline,
                        "phase": phase,
                        "metric": metric,
                        "exp1_best": round(e1_best, 4) if e1_best is not None else None,
                        "exp2_best": round(e2_best, 4) if e2_best is not None else None,
                        "delta_best": round(e2_best - e1_best, 4) if e1_best is not None and e2_best is not None else None,
                        "exp1_mean": round(e1_mean, 4) if e1_mean is not None else None,
                        "exp2_mean": round(e2_mean, 4) if e2_mean is not None else None,
                        "delta_mean": round(e2_mean - e1_mean, 4) if e1_mean is not None and e2_mean is not None else None,
                    })

        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            comp_df.to_csv(f"{output_path}/comparison.csv", index=False)
            comp_df.to_excel(f"{output_path}/comparison.xlsx", index=False, engine='openpyxl')
            logger.info(f"Comparison: {len(comp_df)} rows saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 6: Per-label thresholds (exp2)
    # ═══════════════════════════════════════════════════════════════
    threshold_rows = []
    for _, run in df[(df["experiment"] == "exp2") & (df["metric_phase"] == "val")].iterrows():
        run_name = run.get("run_name", "")
        for col in df.columns:
            if col.startswith("metric_threshold_"):
                label_name = col.replace("metric_threshold_", "")
                threshold_rows.append({
                    "run_name": run_name,
                    "pipeline": run["pipeline"],
                    "label_name": label_name,
                    "optimal_threshold": run[col],
                    "default_threshold": 0.5,
                })

    if threshold_rows:
        thresh_df = pd.DataFrame(threshold_rows)
        thresh_df.to_csv(f"{output_path}/thresholds.csv", index=False)
        thresh_df.to_excel(f"{output_path}/thresholds.xlsx", index=False, engine='openpyxl')
        logger.info(f"Thresholds: {len(thresh_df)} rows saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 7: Per-label F1 (all runs)
    # ═══════════════════════════════════════════════════════════════
    per_label_rows = []
    for _, run in df.iterrows():
        run_name = run.get("run_name", "")
        experiment = run["experiment"]
        pipeline = run["pipeline"]
        phase = run["metric_phase"]

        for col in df.columns:
            if col.startswith("metric_") and col.endswith(")"):
                continue
            if col.startswith("metric_f1_") and not col.startswith("metric_f1_macro") and \
               not col.startswith("metric_f1_micro") and not col.startswith("metric_f1_weighted") and \
               not col.startswith("metric_f1_basic"):
                label_name = col.replace("metric_f1_", "")
                if label_name.startswith("val_") or label_name.startswith("test_"):
                    continue

                per_label_rows.append({
                    "pipeline": pipeline,
                    "run_name": run_name,
                    "experiment": experiment,
                    "phase": phase,
                    "label_name": label_name,
                    "f1_score": run[col],
                })

    if per_label_rows:
        pl_df = pd.DataFrame(per_label_rows)
        pl_df.to_csv(f"{output_path}/per_label_f1.csv", index=False)
        pl_df.to_excel(f"{output_path}/per_label_f1.xlsx", index=False, engine='openpyxl')
        logger.info(f"Per-label F1: {len(pl_df)} rows saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 8: Per-label comparison exp1 vs exp2
    # ═══════════════════════════════════════════════════════════════
    if per_label_rows:
        pl_df = pd.DataFrame(per_label_rows)
        e1_pl = pl_df[pl_df["experiment"] == "exp1"].groupby(["pipeline", "label_name"])["f1_score"].max().reset_index()
        e1_pl.columns = ["pipeline", "label_name", "exp1_f1"]
        e2_pl = pl_df[pl_df["experiment"] == "exp2"].groupby(["pipeline", "label_name"])["f1_score"].max().reset_index()
        e2_pl.columns = ["pipeline", "label_name", "exp2_f1"]

        pl_comp = e1_pl.merge(e2_pl, on=["pipeline", "label_name"], how="outer")
        pl_comp["delta_f1"] = pl_comp["exp2_f1"] - pl_comp["exp1_f1"]
        pl_comp = pl_comp.sort_values("delta_f1", ascending=False)

        if not threshold_rows:
            thresh_df = pd.DataFrame()
        else:
            thresh_df = pd.DataFrame(threshold_rows)

        if not thresh_df.empty:
            avg_thresh = thresh_df.groupby("label_name")["optimal_threshold"].mean().reset_index()
            avg_thresh.columns = ["label_name", "avg_exp2_threshold"]
            pl_comp = pl_comp.merge(avg_thresh, on="label_name", how="left")
            pl_comp["default_threshold"] = 0.5

        pl_comp.to_csv(f"{output_path}/per_label_comparison.csv", index=False)
        pl_comp.to_excel(f"{output_path}/per_label_comparison.xlsx", index=False, engine='openpyxl')
        logger.info(f"Per-label comparison: {len(pl_comp)} rows saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 9: Per-sample detail (from analysis CSVs)
    # ═══════════════════════════════════════════════════════════════
    sample_rows = []
    for analysis_dir, experiment in [("analysis", "exp1"), ("analysis_exp2", "exp2")]:
        if not os.path.exists(analysis_dir):
            continue
        for fname in os.listdir(analysis_dir):
            if not fname.endswith(".csv") or fname.startswith("final_test"):
                continue
            fpath = os.path.join(analysis_dir, fname)
            try:
                adf = pd.read_csv(fpath)
                run_tag = fname.replace("val_analysis_", "").replace(".csv", "")

                def infer_pipeline(tag):
                    tag_lower = tag.lower()
                    if any(x in tag_lower for x in ["bilstm", "cnn"]):
                        return "dl"
                    elif any(x in tag_lower for x in ["indobert", "bert", "xlm", "mmbert"]):
                        return "transformers"
                    return "traditional"

                def infer_target(tag):
                    if "_basic" in tag:
                        return "basic"
                    elif "_fine" in tag:
                        return "fine"
                    return "unknown"

                pipeline = infer_pipeline(run_tag)
                target = infer_target(run_tag)

                for _, row in adf.iterrows():
                    sample_rows.append({
                        "experiment": experiment,
                        "pipeline": pipeline,
                        "run_name": run_tag,
                        "target_level": target,
                        "text": str(row.get("Text", ""))[:200],
                        "true_basic": row.get("True_Basic", ""),
                        "true_fine": row.get("True_Fine", ""),
                        "pred_basic": row.get("Pred_Basic", ""),
                        "pred_fine": row.get("Pred_Fine", ""),
                        "status": row.get("Status", ""),
                    })
            except Exception as e:
                logger.warning(f"Failed to read {fpath}: {e}")

    if sample_rows:
        samp_df = pd.DataFrame(sample_rows)
        samp_df.to_csv(f"{output_path}/per_sample_detail.csv", index=False)
        logger.info(f"Per-sample detail: {len(samp_df)} rows saved (CSV only — too large for Excel)")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 10: Per-sample Hamming Loss & EMR by label combination
    # ═══════════════════════════════════════════════════════════════
    if sample_rows:
        samp_df = pd.DataFrame(sample_rows)
        combo_rows = []
        for (experiment, pipeline, run_name, target), group in samp_df.groupby(["experiment", "pipeline", "run_name", "target_level"]):
            total = len(group)
            exact = len(group[group["status"] == "Exact Match"])
            emr = exact / total if total > 0 else 0
            partial = len(group[group["status"].str.contains("Partial", na=False)])
            mismatch = len(group[group["status"] == "Complete Mismatch"])

            combo_rows.append({
                "experiment": experiment,
                "pipeline": pipeline,
                "run_name": run_name,
                "target_level": target,
                "total_samples": total,
                "exact_match_count": exact,
                "partial_match_count": partial,
                "complete_mismatch_count": mismatch,
                "emr": round(emr, 4),
            })

        if combo_rows:
            combo_df = pd.DataFrame(combo_rows)
            combo_df.to_csv(f"{output_path}/emr_per_run.csv", index=False)
            combo_df.to_excel(f"{output_path}/emr_per_run.xlsx", index=False, engine='openpyxl')
            logger.info(f"EMR per run: {len(combo_df)} rows saved")

    # ═══════════════════════════════════════════════════════════════
    # SHEET 11: F1 per basic emotion group
    # ═══════════════════════════════════════════════════════════════
    if per_label_rows:
        pl_df_full = pd.DataFrame(per_label_rows)
        fine_to_basic = {}
        from src.data_loader import prepare_data
        try:
            _, _, _, _, _, _, _, _, _, _, FINE_TO_ID, _, FINE_TO_BASIC_TAXONOMY = prepare_data("data/new_all.json")
            fine_to_basic = FINE_TO_BASIC_TAXONOMY
        except Exception:
            pass

        if fine_to_basic:
            group_rows = []
            for _, row in pl_df_full.iterrows():
                label = row["label_name"]
                parent = fine_to_basic.get(label, label)
                group_rows.append({**row, "basic_group": parent})

            grp_df = pd.DataFrame(group_rows)
            grp_agg = grp_df.groupby(["pipeline", "experiment", "phase", "basic_group"]).agg(
                mean_f1=("f1_score", "mean"),
                max_f1=("f1_score", "max"),
                min_f1=("f1_score", "min"),
                num_fine_labels=("f1_score", "count"),
            ).reset_index()
            grp_agg = grp_agg.sort_values(["pipeline", "experiment", "mean_f1"], ascending=[True, True, False])

            grp_agg.to_csv(f"{output_path}/f1_per_basic_group.csv", index=False)
            grp_agg.to_excel(f"{output_path}/f1_per_basic_group.xlsx", index=False, engine='openpyxl')
            logger.info(f"F1 per basic group: {len(grp_agg)} rows saved")

    # ═══════════════════════════════════════════════════════════════
    # MASTER EXCEL: All sheets in one file
    # ═══════════════════════════════════════════════════════════════
    master_path = f"{output_path}/experiment_comparison.xlsx"
    with pd.ExcelWriter(master_path, engine='openpyxl') as writer:
        sheets_added = 0

        for exp_tag in ["exp1", "exp2"]:
            csv_path = f"{output_path}/{exp_tag}_val_summary.csv"
            if os.path.exists(csv_path):
                pd.read_csv(csv_path).to_excel(writer, sheet_name=f"{exp_tag}_val", index=False)
                sheets_added += 1

        for exp_tag in ["exp1", "exp2"]:
            csv_path = f"{output_path}/{exp_tag}_test_summary.csv"
            if os.path.exists(csv_path):
                pd.read_csv(csv_path).to_excel(writer, sheet_name=f"{exp_tag}_test", index=False)
                sheets_added += 1

        csv_path = f"{output_path}/comparison.csv"
        if os.path.exists(csv_path):
            pd.read_csv(csv_path).to_excel(writer, sheet_name="comparison", index=False)
            sheets_added += 1

        csv_path = f"{output_path}/thresholds.csv"
        if os.path.exists(csv_path):
            pd.read_csv(csv_path).to_excel(writer, sheet_name="thresholds", index=False)
            sheets_added += 1

        csv_path = f"{output_path}/per_label_f1.csv"
        if os.path.exists(csv_path):
            pdf = pd.read_csv(csv_path)
            if len(pdf) <= 100000:
                pdf.to_excel(writer, sheet_name="per_label_f1", index=False)
            sheets_added += 1

        csv_path = f"{output_path}/per_label_comparison.csv"
        if os.path.exists(csv_path):
            pd.read_csv(csv_path).to_excel(writer, sheet_name="per_label_comparison", index=False)
            sheets_added += 1

        csv_path = f"{output_path}/emr_per_run.csv"
        if os.path.exists(csv_path):
            pd.read_csv(csv_path).to_excel(writer, sheet_name="emr_per_run", index=False)
            sheets_added += 1

        csv_path = f"{output_path}/f1_per_basic_group.csv"
        if os.path.exists(csv_path):
            pd.read_csv(csv_path).to_excel(writer, sheet_name="f1_per_basic_group", index=False)
            sheets_added += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"MASTER EXCEL saved: {master_path}")
    logger.info(f"Total sheets: {sheets_added}")
    logger.info(f"Individual CSVs also in: {output_path}/")
    logger.info(f"{'='*60}")

    return master_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:8002")
    parser.add_argument("--output", type=str, default="experiment_comparison")
    args = parser.parse_args()

    generate_comparison(tracking_uri=args.mlflow_uri, output_path=args.output)
