import os
import argparse
import numpy as np
import pandas as pd
import torch
import mlflow
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from loguru import logger

from src.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary
from src.utils.threshold_tuning import optimize_thresholds, apply_thresholds, log_thresholds
from src.utils.checkpoint import CheckpointManager
from src.utils.mlflow_utils import safe_set_experiment, safe_start_run, safe_end_run, safe_log_param, safe_log_params, safe_log_metric, safe_log_metrics, safe_set_tag
from src.utils.experiment_config import get_config


def train_transformers(data_path, config=None):
    if config is None:
        config = get_config("exp1")

    logger.info("Loading data...")
    logger.info(f"Experiment: {config['experiment']}")
    ckpt = CheckpointManager(config["checkpoint_tf"])
    analysis_dir = config["analysis_dir"]
    saved_models_dir = config["saved_models_dir"]

    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    ) = prepare_data(data_path, preprocessing_mode='transformers')

    models_to_train = {
        "IndoBERT": "indobenchmark/indobert-base-p1",
        "IndoBERT-LEM": "indolem/indobert-base-uncased",
        "IndoBERTweet": "indolem/indobertweet-base-uncased",
        "XLM-R": "xlm-roberta-base",
        "mmBERT": "jhu-clsp/mmBERT-base",
    }

    learning_rates = config["tf_learning_rates"]
    batch_sizes = config["tf_batch_sizes"]
    num_epochs = config["tf_epochs"]
    max_len = config["tf_max_len"]
    target_levels = ['basic', 'fine']

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: EXPERIMENTATION — val metrics only, NO test
    # ═══════════════════════════════════════════════════════════════
    safe_set_experiment(config["mlflow_experiment_p1_tf"])
    phase1_results = ckpt.get_phase1_results()
    logger.info("PHASE 1: Experimentation runs (val metrics only)...")

    for model_name, model_id in models_to_train.items():
        logger.info(f"--- Starting model: {model_name} ({model_id}) ---")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            continue

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding=False, truncation=True, max_length=max_len)

        for target in target_levels:
            is_basic = (target == 'basic')
            num_labels = len(BASIC_TO_ID) if is_basic else len(FINE_TO_ID)
            id_to_label = ID_TO_BASIC if is_basic else ID_TO_FINE
            label_to_id = BASIC_TO_ID if is_basic else FINE_TO_ID
            class_to_parent = {} if is_basic else FINE_TO_BASIC_TAXONOMY
            y_train_labels = (y_train_basic if is_basic else y_train_fine).astype(np.float32).tolist()
            y_val_labels = (y_val_basic if is_basic else y_val_fine).astype(np.float32).tolist()

            logger.info(f"  Target level: {target} ({num_labels} labels)")

            train_dataset = HFDataset.from_dict({
                "text": train_df["preprocessed_text"].tolist(),
                "labels": y_train_labels,
            })
            val_dataset = HFDataset.from_dict({
                "text": val_df["preprocessed_text"].tolist(),
                "labels": y_val_labels,
            })

            tokenized_train = train_dataset.map(tokenize_function, batched=True)
            tokenized_val = val_dataset.map(tokenize_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            def compute_metrics_hf(eval_pred, _id2label=id_to_label, _parent=class_to_parent):
                logits, labels = eval_pred
                probs = torch.sigmoid(torch.tensor(logits)).numpy()
                preds = (probs >= 0.5).astype(np.int32)
                return compute_all_metrics_binary(labels, preds, _id2label, _parent)

            for lr in learning_rates:
                for bs in batch_sizes:
                    hpo_run_name = f"{model_name}_{target}_lr{lr}_bs{bs}"

                    if ckpt.is_completed(hpo_run_name):
                        logger.info(f"    Skipping {hpo_run_name} (already completed)")
                        continue

                    logger.info(f"    Phase 1: {hpo_run_name}")

                    try:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_id,
                            num_labels=num_labels,
                            problem_type="multi_label_classification",
                            id2label=id_to_label,
                            label2id=label_to_id,
                        )
                    except Exception as e:
                        logger.error(f"    Failed to load model for {hpo_run_name}: {e}")
                        continue

                    callbacks = []
                    if config["tf_patience"] is not None:
                        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config["tf_patience"]))

                    training_args = TrainingArguments(
                        output_dir=f"./results/{hpo_run_name}",
                        learning_rate=lr,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs,
                        num_train_epochs=num_epochs,
                        weight_decay=0.01,
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_f1_macro",
                        logging_strategy="epoch",
                        report_to="none",
                        run_name=hpo_run_name,
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_train,
                        eval_dataset=tokenized_val,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics_hf,
                        callbacks=callbacks,
                    )

                    active_run = safe_start_run(run_name=hpo_run_name)
                    mlflow_active = active_run is not None
                    if mlflow_active:
                        safe_set_tag("phase", "experimentation")
                        safe_log_param("model_id", model_id)
                        safe_log_param("target_level", target)
                        safe_log_param("learning_rate", lr)
                        safe_log_param("batch_size", bs)
                        safe_log_param("num_epochs", num_epochs)
                        safe_log_param("max_len", max_len)
                        safe_log_param("threshold_tuning", config["threshold_tuning"])

                    trainer.train()

                    val_results = trainer.evaluate(tokenized_val, metric_key_prefix="val")
                    val_f1_macro = val_results.get("val_f1_macro", -1.0)

                    if mlflow_active:
                        for key, value in val_results.items():
                            if key.startswith("val_") and isinstance(value, (int, float)):
                                safe_log_metric(key, value)

                    preds_output = trainer.predict(tokenized_val)
                    val_logits = preds_output.predictions
                    val_probs = torch.sigmoid(torch.tensor(val_logits)).numpy()
                    y_val_true = preds_output.label_ids

                    thresholds = None
                    if config["threshold_tuning"]:
                        thresholds = optimize_thresholds(y_val_true, val_probs, id_to_label)
                        log_thresholds(thresholds, id_to_label, mlflow_log_fn=safe_log_metric)
                        y_val_pred = apply_thresholds(val_probs, thresholds)
                    else:
                        y_val_pred = (val_probs >= 0.5).astype(np.int32)

                    os.makedirs(analysis_dir, exist_ok=True)
                    val_analysis_path = f"{analysis_dir}/val_analysis_{hpo_run_name}.csv"
                    save_manual_analysis_binary(
                        val_df["text"].tolist(),
                        y_val_true,
                        y_val_pred,
                        id_to_label,
                        class_to_parent,
                        val_analysis_path,
                    )

                    phase1_results.append({
                        'model_name': model_name,
                        'model_id': model_id,
                        'target_level': target,
                        'learning_rate': lr,
                        'batch_size': bs,
                        'val_f1_macro': val_f1_macro,
                        'thresholds': thresholds,
                    })
                    ckpt.add_phase1_result({
                        'model_name': model_name,
                        'model_id': model_id,
                        'target_level': target,
                        'learning_rate': lr,
                        'batch_size': bs,
                        'val_f1_macro': float(val_f1_macro),
                        'thresholds': thresholds,
                    })
                    ckpt.mark_completed(hpo_run_name)

                    logger.info(f"    {hpo_run_name} | Val F1-Macro: {val_f1_macro:.4f}")

                    save_dir = f"{saved_models_dir}/{hpo_run_name}"
                    os.makedirs(save_dir, exist_ok=True)
                    trainer.save_model(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    if thresholds:
                        np.save(f"{save_dir}/thresholds.npy", thresholds)
                    logger.info(f"    Model saved to {save_dir}/")

                    if mlflow_active:
                        safe_end_run()

                    del model, trainer
                    torch.cuda.empty_cache()

        logger.info(f"  All Phase 1 HPO runs for {model_name} completed.")

    logger.info(f"\nPHASE 1 complete. {len(phase1_results)} runs logged (val metrics only).")
    ckpt.mark_phase_complete(1)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: FINAL REPORT — retrain best on train+val, test ONCE
    # ═══════════════════════════════════════════════════════════════
    phase1_results = ckpt.get_phase1_results()
    if not phase1_results:
        logger.error("No Phase 1 results found. Cannot proceed to Phase 2.")
        return
    safe_set_experiment(config["mlflow_experiment_p2_tf"])
    logger.info("\nPHASE 2: Retraining best models on train+val, evaluating on test (ONCE per model+target)...")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    test_texts_raw = test_df["text"].tolist()

    for model_name, model_id in models_to_train.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Phase 2: Failed to load tokenizer for {model_name}: {e}")
            continue

        def tokenize_function_tv(examples):
            return tokenizer(examples["text"], padding=False, truncation=True, max_length=max_len)

        for target in target_levels:
            model_target_results = [
                r for r in phase1_results
                if r['model_name'] == model_name and r['target_level'] == target
            ]
            if not model_target_results:
                logger.warning(f"No Phase 1 results for {model_name}/{target}, skipping.")
                continue

            best = max(model_target_results, key=lambda x: x['val_f1_macro'])
            best_thresholds = best.get('thresholds')
            logger.info(f"Best {model_name}/{target}: lr={best['learning_rate']}, bs={best['batch_size']} "
                         f"(val F1-Macro: {best['val_f1_macro']:.4f})")

            is_basic = (target == 'basic')
            num_labels = len(BASIC_TO_ID) if is_basic else len(FINE_TO_ID)
            id_to_label = ID_TO_BASIC if is_basic else ID_TO_FINE
            label_to_id = BASIC_TO_ID if is_basic else FINE_TO_ID
            class_to_parent = {} if is_basic else FINE_TO_BASIC_TAXONOMY

            y_train_val_basic = np.concatenate([y_train_basic, y_val_basic], axis=0).astype(np.float32)
            y_train_val_fine = np.concatenate([y_train_fine, y_val_fine], axis=0).astype(np.float32)
            y_train_val_labels = (y_train_val_basic if is_basic else y_train_val_fine).tolist()
            y_test_labels = (y_test_basic if is_basic else y_test_fine).astype(np.float32).tolist()

            train_val_dataset = HFDataset.from_dict({
                "text": train_val_df["preprocessed_text"].tolist(),
                "labels": y_train_val_labels,
            })
            test_dataset = HFDataset.from_dict({
                "text": test_df["preprocessed_text"].tolist(),
                "labels": y_test_labels,
            })

            tokenized_train_val = train_val_dataset.map(tokenize_function_tv, batched=True)
            tokenized_test = test_dataset.map(tokenize_function_tv, batched=True)

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    num_labels=num_labels,
                    problem_type="multi_label_classification",
                    id2label=id_to_label,
                    label2id=label_to_id,
                )
            except Exception as e:
                logger.error(f"Phase 2: Failed to load model for {model_name}/{target}: {e}")
                continue

            final_run_name = f"FINAL_{model_name}_{target}_lr{best['learning_rate']}_bs{best['batch_size']}"

            if ckpt.is_completed(final_run_name):
                logger.info(f"Skipping {final_run_name} (already completed)")
                continue

            training_args = TrainingArguments(
                output_dir=f"./results/{final_run_name}",
                learning_rate=best['learning_rate'],
                per_device_train_batch_size=best['batch_size'],
                per_device_eval_batch_size=best['batch_size'],
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                eval_strategy="no",
                save_strategy="no",
                logging_strategy="epoch",
                report_to="none",
                run_name=final_run_name,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_val,
                data_collator=data_collator,
            )

            active_run = safe_start_run(run_name=final_run_name)
            mlflow_active = active_run is not None
            if mlflow_active:
                safe_set_tag("phase", "final_test")
                safe_log_params({
                    "model_id": model_id,
                    "target_level": target,
                    "learning_rate": best['learning_rate'],
                    "batch_size": best['batch_size'],
                    "num_epochs": num_epochs,
                    "max_len": max_len,
                    "selected_by_val_f1_macro": best['val_f1_macro'],
                    "threshold_tuning": config["threshold_tuning"],
                })

            logger.info(f"Phase 2: Fine-tuning {final_run_name} on train+val ({len(train_val_df)} samples)...")
            trainer.train()

            logger.info(f"Phase 2: Predicting on test set...")
            preds_output = trainer.predict(tokenized_test)
            logits = preds_output.predictions
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            y_true_binary = preds_output.label_ids

            if config["threshold_tuning"] and best_thresholds:
                y_pred_binary = apply_thresholds(probs, best_thresholds)
                logger.info(f"Using optimized thresholds from Phase 1")
            else:
                y_pred_binary = (probs >= 0.5).astype(np.int32)

            test_metrics = compute_all_metrics_binary(y_true_binary, y_pred_binary, id_to_label, class_to_parent)
            if mlflow_active:
                for k, v in test_metrics.items():
                    safe_log_metric(f"test_{k}", v)

            os.makedirs(analysis_dir, exist_ok=True)
            analysis_path = f"{analysis_dir}/final_test_analysis_{final_run_name}.csv"
            save_manual_analysis_binary(
                test_texts_raw,
                y_true_binary,
                y_pred_binary,
                id_to_label,
                class_to_parent,
                analysis_path,
            )

            logger.info(f"FINAL {final_run_name} | Test F1-Macro: {test_metrics['f1_macro']:.4f} | "
                        f"Subset Acc: {test_metrics['subset_accuracy']:.4f} | "
                        f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")

            if mlflow_active:
                try:
                    mlflow.transformers.log_model(trainer.model, "model")
                except Exception as e:
                    logger.warning(f"Failed to log model artifact: {e}")
                safe_end_run()

            save_dir = f"{saved_models_dir}/{final_run_name}"
            os.makedirs(save_dir, exist_ok=True)
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)
            if best_thresholds:
                np.save(f"{save_dir}/thresholds.npy", best_thresholds)
            logger.info(f"Model saved to {save_dir}/")

            ckpt.mark_completed(final_run_name)

            del model, trainer
            torch.cuda.empty_cache()

    ckpt.mark_phase_complete(2)
    logger.info("\n=== Transformer pipeline complete ===")
    logger.info(f"Phase 1: 30 runs with val metrics → {analysis_dir}/val_analysis_*.csv, {saved_models_dir}/")
    logger.info(f"Phase 2: 10 final runs with test metrics → {analysis_dir}/final_test_analysis_*.csv, {saved_models_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    import sys
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

    train_transformers(args.data_path)
