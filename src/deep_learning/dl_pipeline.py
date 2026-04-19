import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import argparse
from loguru import logger
import numpy as np

import os

from src.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics, save_manual_analysis
from src.deep_learning.models import TextDataset, BiLSTM, TextCNN

def train_dl(data_path: str):
    logger.info("Loading Data...")
    train_df, val_df, test_df, class_to_id, id_to_class, fine_to_basic = prepare_data(data_path)
    
    logger.info("Initializing Datasets...")
    train_dataset = TextDataset(train_df['text'].values, train_df['label_id'].values, max_len=128)
    vocab = train_dataset.vocab
    
    val_dataset = TextDataset(val_df['text'].values, val_df['label_id'].values, vocab=vocab, max_len=128)
    test_dataset = TextDataset(test_df['text'].values, test_df['label_id'].values, vocab=vocab, max_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    vocab_size = len(vocab)
    num_classes = len(class_to_id)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # ABLATION STUDY CONFIGURATION
    # Modify these lists to automatically iterate and test 
    # different conditions in Deep Learning!
    # ---------------------------------------------------------
    ablation_params = {
        "embed_dims": [100, 300], # Test different embedding sizes
        "hidden_dims": [64, 128], # Test different LSTM specific hidden states
    }
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5
    
    mlflow.set_experiment("Deep_Learning_Ablation")
    
    # Loop over ablation parameters
    for embed_dim in ablation_params["embed_dims"]:
        for hidden_dim in ablation_params["hidden_dims"]:
            
            models = {
                "Bi-LSTM": BiLSTM(vocab_size, embed_dim, hidden_dim, num_classes).to(device),
                "CNN": TextCNN(vocab_size, embed_dim, num_classes).to(device)
            }
            
            for model_name, model in models.items():
                run_tag = f"{model_name}_emb{embed_dim}_hid{hidden_dim}"
                logger.info(f"--- Running Ablation: {run_tag} ---")
                
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                
                with mlflow.start_run(run_name=run_tag):
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("vocab_size", vocab_size)
                    mlflow.log_param("embed_dim", embed_dim)
                    mlflow.log_param("batch_size", 32)
                    mlflow.log_param("epochs", num_epochs)
                    if model_name == "Bi-LSTM":
                        mlflow.log_param("hidden_dim", hidden_dim)
                
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for texts, labels in train_loader:
                    texts, labels = texts.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    
                avg_train_loss = total_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                
                # Validation evaluation
                model.eval()
                all_preds = []
                all_labels = []
                val_loss = 0
                with torch.no_grad():
                    for texts, labels in val_loader:
                        texts, labels = texts.to(device), labels.to(device)
                        outputs = model(texts)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                metrics = compute_all_metrics(all_labels, all_preds, id_to_class, fine_to_basic)
                mlflow.log_metric("val_loss", val_loss / len(val_loader), step=epoch)
                for k, v in metrics.items():
                    mlflow.log_metric(f"val_{k}", v, step=epoch)
                    
            # Test evaluation
            if len(test_loader) > 0:
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for texts, labels in test_loader:
                        texts, labels = texts.to(device), labels.to(device)
                        outputs = model(texts)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                metrics = compute_all_metrics(all_labels, all_preds, id_to_class, fine_to_basic)
                for k, v in metrics.items():
                    mlflow.log_metric(f"test_{k}", v)

                # Save manual analysis file
                if not os.path.exists("analysis"):
                    os.makedirs("analysis")
                analysis_path = f"analysis/manual_analysis_{run_tag}.csv"
                save_manual_analysis(test_df['text'].values, all_labels, all_preds, id_to_class, fine_to_basic, analysis_path)
                logger.info(f"Manual analysis exported to: {analysis_path}")

            # mlflow.pytorch.log_model(model, "model")
            logger.info(f"{model_name} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    import sys
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")
    
    train_dl(args.data_path)
