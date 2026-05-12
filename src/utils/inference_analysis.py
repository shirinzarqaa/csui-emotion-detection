import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime


def run_inference_and_analysis(
    y_pred,
    y_prob,
    y_test,
    texts,
    id_to_label,
    fine_to_basic=None,
    pipeline_name="",
    output_dir="analysis",
):
    """
    Deep inference analysis per sample on test set.
    Called ONCE at Phase 2 (final report).

    y_pred: (N, C) binary predictions
    y_prob: (N, C) probability scores (sigmoid output)
    y_test: (N, C) binary ground truth
    texts: list of raw texts
    id_to_label: dict mapping idx -> label name
    fine_to_basic: taxonomy dict (empty for basic level)
    pipeline_name: string identifier for output files
    output_dir: directory for CSV output
    """
    os.makedirs(output_dir, exist_ok=True)
    n_samples, n_labels = y_test.shape
    label_names = [id_to_label[i] for i in range(n_labels)]

    records = []
    for i in range(n_samples):
        true_indices = set(np.where(y_test[i] == 1)[0])
        pred_indices = set(np.where(y_pred[i] == 1)[0])

        true_labels = sorted([label_names[j] for j in true_indices])
        pred_labels = sorted([label_names[j] for j in pred_indices])

        if len(pred_indices) > 0:
            confidence = float(np.mean([y_prob[i, j] for j in pred_indices]))
        else:
            confidence = 0.0

        n_correct = len(true_indices & pred_indices)
        n_overpredict = len(pred_indices - true_indices)
        n_underpredict = len(true_indices - pred_indices)

        if true_indices == pred_indices:
            status = "exact_match"
        elif len(true_indices & pred_indices) > 0:
            status = f"partial_{len(true_indices & pred_indices)}_overlap"
        else:
            status = "complete_mismatch"

        ambiguous = [
            label_names[j] for j in range(n_labels)
            if 0.3 <= y_prob[i, j] <= 0.7
        ]

        true_basic = sorted(set(
            fine_to_basic.get(l, l) for l in true_labels
        )) if fine_to_basic else true_labels
        pred_basic = sorted(set(
            fine_to_basic.get(l, l) for l in pred_labels
        )) if fine_to_basic else pred_labels

        record = {
            "sample_id": i,
            "text": texts[i] if i < len(texts) else "",
            "true_labels": ",".join(true_labels) if true_labels else "none",
            "pred_labels": ",".join(pred_labels) if pred_labels else "none",
            "true_basic": ",".join(true_basic) if true_basic else "none",
            "pred_basic": ",".join(pred_basic) if pred_basic else "none",
            "confidence": round(confidence, 4),
            "n_true": len(true_labels),
            "n_pred": len(pred_labels),
            "n_correct": n_correct,
            "n_overpredict": n_overpredict,
            "n_underpredict": n_underpredict,
            "status": status,
            "ambiguous_labels": ",".join(ambiguous) if ambiguous else "none",
        }
        for j in range(n_labels):
            record[f"prob_{label_names[j]}"] = round(float(y_prob[i, j]), 4)

        records.append(record)

    df = pd.DataFrame(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"deep_analysis_{pipeline_name}_{timestamp}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    xlsx_path = os.path.join(output_dir, f"deep_analysis_{pipeline_name}_{timestamp}.xlsx")
    df.to_excel(xlsx_path, index=False, engine='openpyxl')

    summary_path = os.path.join(output_dir, f"deep_analysis_summary_{pipeline_name}_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Pipeline: {pipeline_name}\n")
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Labels: {n_labels}\n\n")
        f.write("Status distribution:\n")
        for status, count in df['status'].value_counts().items():
            f.write(f"  {status}: {count}\n")
        f.write(f"\nAvg confidence (exact_match): {df[df['status']=='exact_match']['confidence'].mean():.4f}\n")
        mismatch_mask = df['status'].str.startswith('complete_mismatch')
        if mismatch_mask.any():
            f.write(f"Avg confidence (complete_mismatch): {df[mismatch_mask]['confidence'].mean():.4f}\n")
        f.write(f"\nSamples with low confidence (<0.55): {len(df[df['confidence'] < 0.55])}\n")
        f.write(f"Samples with ambiguous labels: {len(df[df['ambiguous_labels'] != 'none'])}\n")
        f.write(f"\nPer-label average probability:\n")
        for j in range(n_labels):
            avg_prob = df[f"prob_{label_names[j]}"].mean()
            f.write(f"  {label_names[j]}: {avg_prob:.4f}\n")

    hard_mask = (
        df['status'].str.startswith('complete_mismatch') |
        (df['confidence'] < 0.55) |
        (df['ambiguous_labels'] != 'none')
    )
    hard_df = df[hard_mask].sort_values('confidence')
    hard_path = os.path.join(output_dir, f"hard_samples_{pipeline_name}_{timestamp}.csv")
    hard_df.to_csv(hard_path, index=False, encoding='utf-8')
    hard_xlsx = os.path.join(output_dir, f"hard_samples_{pipeline_name}_{timestamp}.xlsx")
    hard_df.to_excel(hard_xlsx, index=False, engine='openpyxl')

    print(f"Deep analysis: {csv_path}")
    print(f"Hard samples:  {hard_path} ({len(hard_df)} samples)")
    print(f"XLSX:          {xlsx_path}")
    print(f"Summary:       {summary_path}")

    return df


def compare_across_pipelines(
    dfs, pipeline_names, output_dir="analysis"
):
    """
    Compare predictions across multiple pipelines for the same test set.
    Identify samples where all pipelines agree/disagree.
    """
    n = len(dfs[0])
    records = []
    label_cols = [c for c in dfs[0].columns if c.startswith('prob_')]

    for i in range(n):
        row = {"sample_id": i, "text": dfs[0].iloc[i]['text']}

        statuses = []
        confidences = []
        for df_idx, (df, name) in enumerate(zip(dfs, pipeline_names)):
            row[f"{name}_pred"] = df.iloc[i]['pred_labels']
            row[f"{name}_true"] = df.iloc[i]['true_labels']
            row[f"{name}_status"] = df.iloc[i]['status']
            row[f"{name}_confidence"] = df.iloc[i]['confidence']
            statuses.append(df.iloc[i]['status'])
            confidences.append(df.iloc[i]['confidence'])

        row['true_labels'] = dfs[0].iloc[i]['true_labels']
        row['all_exact'] = all(s == 'exact_match' for s in statuses)
        row['all_wrong'] = all(s.startswith('complete_mismatch') for s in statuses)
        row['avg_confidence'] = round(np.mean(confidences), 4)

        records.append(row)

    df_compare = pd.DataFrame(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"cross_pipeline_comparison_{timestamp}.csv")
    df_compare.to_csv(path, index=False, encoding='utf-8')
    xlsx_path = os.path.join(output_dir, f"cross_pipeline_comparison_{timestamp}.xlsx")
    df_compare.to_excel(xlsx_path, index=False, engine='openpyxl')

    print(f"\nCross-pipeline comparison: {path}")
    print(f"  All exact match:   {df_compare['all_exact'].sum()}")
    print(f"  All complete miss: {df_compare['all_wrong'].sum()}")

    return df_compare
