import os
import argparse
import mlflow
import torch
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import os
from loguru import logger

from src.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics, save_manual_analysis

def train_transformers(data_path: str):
    logger.info("Loading Data...")
    train_df, val_df, test_df, class_to_id, id_to_class, fine_to_basic = prepare_data(data_path)
    
    # Closure for metric evaluation to pass dictionary maps
    def compute_metrics_hf(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        
        metrics = compute_all_metrics(labels, predictions, id_to_class, fine_to_basic)
        return metrics
    
    # Standard HF Dataset preparation
    train_dataset = HFDataset.from_pandas(train_df[['text', 'label_id']])
    val_dataset = HFDataset.from_pandas(val_df[['text', 'label_id']])
    test_dataset = HFDataset.from_pandas(test_df[['text', 'label_id']]) if not test_df.empty else None
    
    num_classes = len(class_to_id)
    
    models_to_train = {
        "IndoBERT_IndoNLU": "indobenchmark/indobert-base-p1",
        "IndoBERT_IndoLEM": "indolem/indobert-base-uncased",
        "IndoBERTtweet": "indolem/indobertweet-base-uncased",
        "XLM-R": "xlm-roberta-base",
        "mmBERT": "jhu-clsp/mmBERT-base"
    }
    
    mlflow.set_experiment("Transformer_Models")

    for model_name, model_id in models_to_train.items():
        logger.info(f"Training {model_name} from {model_id}...")
        
        # In case some models error out, we continue
        try:
            # Tokenizer remains the same across hyperparameter variations
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            def tokenize_function(examples):
                return tokenizer(examples["text"], padding=False, truncation=True, max_length=128)

            tokenized_train = train_dataset.map(tokenize_function, batched=True).rename_column("label_id", "labels")
            tokenized_val = val_dataset.map(tokenize_function, batched=True).rename_column("label_id", "labels")
            if test_dataset:
                tokenized_test = test_dataset.map(tokenize_function, batched=True).rename_column("label_id", "labels")

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            # ---------------------------------------------------------
            # HYPERPARAMETER OPTIMIZATION (HPO) GRID
            # Adjust these to find the best performing model configuration!
            # ---------------------------------------------------------
            # Menggunakan standar deviasi learning rate rekomendasi Devlin et al. (2019)
            learning_rates = [2e-5, 3e-5, 4e-5, 5e-5] 
            batch_sizes = [16, 32]        # Evaluates gradient stability
            
            for lr in learning_rates:
                for bs in batch_sizes:
                    hpo_run_name = f"{model_name}_lr{lr}_bs{bs}"
                    logger.info(f"--- Running HPO configuration: {hpo_run_name} ---")
                    
                    # We MUST load a fresh model each iteration to prevent catastrophic logic collision
                    # where models resume training off the previously trained iteration
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_id, 
                        num_labels=num_classes,
                        id2label=id_to_class,
                        label2id=class_to_id
                    )

                    # Hyperparameters based on literature configurations for fine-tuning PLMs
                    training_args = TrainingArguments(
                        output_dir=f"./results/{hpo_run_name}",
                        # Dynamic parameter injection based on loop
                        learning_rate=lr,  
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs,
                        # Moderate epochs to achieve convergence without heavy overfitting
                        num_train_epochs=3,
                        weight_decay=0.01,
                        eval_strategy="epoch",
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        metric_for_best_model="f1_macro",
                        # Log automatically to mlflow
                        report_to="mlflow",
                        run_name=hpo_run_name
                    )
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_train,
                        eval_dataset=tokenized_val,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics_hf,
                    )

                    # Start run to ensure it falls under our custom experiment name cleanly
                    with mlflow.start_run(run_name=hpo_run_name):
                        mlflow.log_param("model_id", model_id)
                        trainer.train()
                        
                        # Evaluate on test set
                        if test_dataset:
                            test_results = trainer.evaluate(tokenized_test, metric_key_prefix="test")
                            logger.info(f"{hpo_run_name} Test Results: {test_results}")

                            # Save Manual Analysis by running predictions via trainer manually
                            preds = trainer.predict(tokenized_test)
                            predicted_labels = preds.predictions.argmax(axis=-1)
                            actual_labels = preds.label_ids
                            
                            if not os.path.exists("analysis"):
                                os.makedirs("analysis")
                            analysis_path = f"analysis/manual_analysis_{hpo_run_name}.csv"
                            save_manual_analysis(test_df['text'].values, actual_labels, predicted_labels, id_to_class, fine_to_basic, analysis_path)
                            logger.info(f"Manual analysis exported to: {analysis_path}")

            logger.info(f"All HPO combinations for {model_name} completed successfully.")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    import sys
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")
    
    train_transformers(args.data_path)
