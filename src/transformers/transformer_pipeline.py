import os
import argparse
import numpy as np
import torch
import mlflow
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from loguru import logger

from src.data_loader import prepare_data
from src.utils.preprocessing import preprocess_for_transformers
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary


def train_transformers(data_path: str):
    logger.info("Loading data...")
    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    ) = prepare_data(data_path)

    logger.info("Preprocessing texts...")
    train_df["preprocessed_text"] = train_df["text"].apply(preprocess_for_transformers)
    val_df["preprocessed_text"] = val_df["text"].apply(preprocess_for_transformers)
    if not test_df.empty:
        test_df["preprocessed_text"] = test_df["text"].apply(preprocess_for_transformers)

    models_to_train = {
        "IndoBERT": "indobenchmark/indobert-base-p1",
        "IndoBERT-LEM": "indolem/indobert-base-uncased",
        "IndoBERTweet": "indolem/indobertweet-base-uncased",
        "XLM-R": "xlm-roberta-base",
        "mmBERT": "jhu-clsp/mmBERT-base",
    }

    mlflow.set_experiment("Transformer_MultiLabel")

    learning_rates = [2e-5, 3e-5, 4e-5, 5e-5]
    batch_sizes = [16, 32]
    num_epochs = 3

    for model_name, model_id in models_to_train.items():
        logger.info(f"--- Starting model: {model_name} ({model_id}) ---")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            continue

        def tokenize_function(examples):
            return tokenizer(examples["preprocessed_text"], padding=False, truncation=True, max_length=128)

        train_labels = y_train_fine.astype(np.float32).tolist()
        val_labels = y_val_fine.astype(np.float32).tolist()

        train_dataset = HFDataset.from_dict({
            "preprocessed_text": train_df["preprocessed_text"].tolist(),
            "labels": train_labels,
        })
        val_dataset = HFDataset.from_dict({
            "preprocessed_text": val_df["preprocessed_text"].tolist(),
            "labels": val_labels,
        })

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)

        tokenized_test = None
        if not test_df.empty:
            test_labels = y_test_fine.astype(np.float32).tolist()
            test_dataset = HFDataset.from_dict({
                "preprocessed_text": test_df["preprocessed_text"].tolist(),
                "labels": test_labels,
            })
            tokenized_test = test_dataset.map(tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        def compute_metrics_hf(eval_pred):
            logits, labels = eval_pred
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs >= 0.5).astype(np.int32)
            return compute_all_metrics_binary(labels, preds, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY)

        for lr in learning_rates:
            for bs in batch_sizes:
                hpo_run_name = f"{model_name}_lr{lr}_bs{bs}"
                logger.info(f"  Running HPO: {hpo_run_name}")

                try:
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_id,
                        num_labels=46,
                        problem_type="multi_label_classification",
                        id2label=ID_TO_FINE,
                        label2id=FINE_TO_ID,
                    )
                except Exception as e:
                    logger.error(f"  Failed to load model for {hpo_run_name}: {e}")
                    continue

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
                    report_to="mlflow",
                    run_name=hpo_run_name,
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_val,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics_hf,
                )

                with mlflow.start_run(run_name=hpo_run_name):
                    mlflow.log_param("model_id", model_id)
                    mlflow.log_param("learning_rate", lr)
                    mlflow.log_param("batch_size", bs)
                    mlflow.log_param("num_epochs", num_epochs)

                    trainer.train()

                    if tokenized_test is not None:
                        test_results = trainer.evaluate(tokenized_test, metric_key_prefix="test")
                        logger.info(f"  {hpo_run_name} test results: {test_results}")

                        with mlflow.start_run(run_name=f"{hpo_run_name}_test", nested=True):
                            for key, value in test_results.items():
                                if key.startswith("test_") and isinstance(value, (int, float)):
                                    mlflow.log_metric(key.replace("test_", ""), value)

                        preds_output = trainer.predict(tokenized_test)
                        logits = preds_output.predictions
                        probs = torch.sigmoid(torch.tensor(logits)).numpy()
                        y_pred_binary = (probs >= 0.5).astype(np.int32)
                        y_true_binary = preds_output.label_ids

                        if not os.path.exists("analysis"):
                            os.makedirs("analysis")
                        analysis_path = f"analysis/manual_analysis_{hpo_run_name}.csv"
                        save_manual_analysis_binary(
                            test_df["text"].tolist(),
                            y_true_binary,
                            y_pred_binary,
                            ID_TO_FINE,
                            FINE_TO_BASIC_TAXONOMY,
                            analysis_path,
                        )
                        logger.info(f"  Manual analysis saved to: {analysis_path}")

        logger.info(f"  All HPO runs for {model_name} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    import sys
    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

    train_transformers(args.data_path)