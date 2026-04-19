import argparse
from loguru import logger
import sys

from src.traditional.traditional_pipeline import train_traditional
from src.deep_learning.dl_pipeline import train_dl
from src.transformers.transformer_pipeline import train_transformers

def main():
    parser = argparse.ArgumentParser(description="10-Model Emission Classification Pipeline Orchestrator")
    parser.add_argument("--data_path", type=str, default="./data/dataset.json", help="Path to input JSON data")
    parser.add_argument("--run", type=str, choices=["traditional", "dl", "transformers", "all"], default="all",
                        help="Which pipeline to run. Choose one: traditional, dl, transformers, all (default)")
                        
    args = parser.parse_args()
    
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")
    
    logger.info(f"Starting execution for {args.run} using data from {args.data_path}")

    if args.run in ["all", "traditional"]:
        logger.info("=== Starting Traditional ML Pipeline ===")
        train_traditional(args.data_path)
        
    if args.run in ["all", "dl"]:
        logger.info("=== Starting PyTorch Deep Learning Pipeline ===")
        train_dl(args.data_path)
        
    if args.run in ["all", "transformers"]:
        logger.info("=== Starting HuggingFace Transformers Pipeline ===")
        train_transformers(args.data_path)
        
    logger.info("Pipeline execution completed successfully. Please check MLflow server for metrics tracked.")

if __name__ == "__main__":
    main()
