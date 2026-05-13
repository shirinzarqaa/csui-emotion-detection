import argparse
import os
from loguru import logger
import sys

import mlflow

from src.traditional.traditional_pipeline import train_traditional
from src.deep_learning.dl_pipeline import train_dl
from src.transformers.transformer_pipeline import train_transformers

def main():
    parser = argparse.ArgumentParser(description="Multi-Label Emotion Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="./data/new_all.json")
    parser.add_argument("--run", type=str, choices=["traditional", "dl", "transformers", "all"], default="all")
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()
    
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")

    tracking_uri = args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8002")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    logger.info(f"Starting execution for {args.run} using data from {args.data_path}")

    if args.run in ["all", "traditional"]:
        logger.info("=== Starting Traditional ML Pipeline ===")
        train_traditional(args.data_path, n_jobs=args.n_jobs)
        
    if args.run in ["all", "dl"]:
        logger.info("=== Starting PyTorch Deep Learning Pipeline ===")
        train_dl(args.data_path)
        
    if args.run in ["all", "transformers"]:
        logger.info("=== Starting HuggingFace Transformers Pipeline ===")
        train_transformers(args.data_path)
        
    logger.info("Pipeline execution completed. Check MLflow UI for metrics.")

if __name__ == "__main__":
    main()
