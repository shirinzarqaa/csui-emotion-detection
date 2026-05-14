import argparse
import os
from loguru import logger
import sys

import mlflow

from src.traditional.traditional_pipeline import train_traditional
from src.deep_learning.dl_pipeline import train_dl
from src.transformers.transformer_pipeline import train_transformers
from src.utils.experiment_config import get_config

def main():
    parser = argparse.ArgumentParser(description="Multi-Label Emotion Classification Pipeline")
    parser.add_argument("--data_path", type=str, default="./data/new_all.json")
    parser.add_argument("--run", type=str, choices=["traditional", "dl", "transformers", "all"], default="all")
    parser.add_argument("--mlflow_uri", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--experiment", type=str, default="exp1", choices=["exp1", "exp2"])
    args = parser.parse_args()
    
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")

    config = get_config(args.experiment)
    
    tracking_uri = args.mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8002")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Config: threshold_tuning={config['threshold_tuning']}, pos_weight={config['pos_weight']}, "
                f"dl_max_len={config['dl_max_len']}, tf_max_len={config['tf_max_len']}, "
                f"tf_epochs={config['tf_epochs']}, dl_patience={config['dl_patience']}")
    
    logger.info(f"Starting execution for {args.run} using data from {args.data_path}")

    if args.run in ["all", "traditional"]:
        logger.info("=== Starting Traditional ML Pipeline ===")
        train_traditional(args.data_path, n_jobs=args.n_jobs, config=config)
        
    if args.run in ["all", "dl"]:
        logger.info("=== Starting PyTorch Deep Learning Pipeline ===")
        train_dl(args.data_path, config=config)
        
    if args.run in ["all", "transformers"]:
        logger.info("=== Starting HuggingFace Transformers Pipeline ===")
        train_transformers(args.data_path, config=config)
        
    logger.info("Pipeline execution completed. Check MLflow UI for metrics.")

if __name__ == "__main__":
    main()
