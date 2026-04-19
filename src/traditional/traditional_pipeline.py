import os
import mlflow
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from loguru import logger 
import os

from src.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics, save_manual_analysis

def train_traditional(data_path: str):
    logger.info("Loading Data...")
    # NOTE ON ADDING DATA:
    # To use new data, simply pass the path to your JSON into this script.
    # The `prepare_data` loader automatically adapts to any new rows or labels!
    train_df, val_df, test_df, class_to_id, id_to_class, fine_to_basic = prepare_data(data_path)
    
    logger.info("Setting up Traditional ML Parameter configurations...")
    
    # ---------------------------------------------------------
    # FEATURE ABLATION CONFIGURATION
    # Modify ngram_ranges or max_features to run tests easily!
    # ---------------------------------------------------------
    feature_configs = [
        {"name": "Unigram", "ngram_range": (1, 1), "max_features": 5000},
        {"name": "Bigram", "ngram_range": (1, 2), "max_features": 10000}
    ]
    
    mlflow.set_experiment("Traditional_ML_Models")

    for feat_conf in feature_configs:
        logger.info(f"Extracting features using {feat_conf['name']} TF-IDF...")
        vectorizer = TfidfVectorizer(ngram_range=feat_conf['ngram_range'], max_features=feat_conf['max_features'])
        
        X_train = vectorizer.fit_transform(train_df['text'])
        y_train = train_df['label_id']
        
        X_val = vectorizer.transform(val_df['text'])
        y_val = val_df['label_id']
        
        X_test = vectorizer.transform(test_df['text']) if not test_df.empty else None
        y_test = test_df['label_id'] if not test_df.empty else None
        
        # Implementing base architectural choices based on thesis references
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            # Explicit alpha=1.0 to ensure Laplace Smoothing representation (Pang et al., 2002)
            "NaiveBayes": MultinomialNB(alpha=1.0),
            # Switching to RBF kernel as proposed optimal for Indonesian text (Barus et al., 2025)
            "SVM": SVC(kernel='rbf', random_state=42)
        }

        for model_name, model in models.items():
            run_tag = f"{model_name}_{feat_conf['name']}"
            logger.info(f"Training {run_tag}...")
            
            with mlflow.start_run(run_name=run_tag):
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("feature_extraction", feat_conf['name'])
                mlflow.log_param("max_features", feat_conf['max_features'])
            
            # Log model's native params
            for param_name, param_val in model.get_params().items():
                # Log model's native params
                for param_name, param_val in model.get_params().items():
                    mlflow.log_param(param_name, param_val)
                
                model.fit(X_train, y_train)
                
                # Evaluate on Validation
                preds_val = model.predict(X_val)
                metrics_val = compute_all_metrics(y_val, preds_val, id_to_class, fine_to_basic)
                for k, v in metrics_val.items():
                    mlflow.log_metric(f"val_{k}", v)
                    
                # Evaluate on Test
                if not test_df.empty:
                    preds_test = model.predict(X_test)
                    metrics_test = compute_all_metrics(y_test, preds_test, id_to_class, fine_to_basic)
                    for k, v in metrics_test.items():
                        mlflow.log_metric(f"test_{k}", v)
                        
                    # Save manual analysis file
                    if not os.path.exists("analysis"):
                        os.makedirs("analysis")
                    analysis_path = f"analysis/manual_analysis_{run_tag}.csv"
                    save_manual_analysis(test_df['text'].values, y_test.values, preds_test, id_to_class, fine_to_basic, analysis_path)
                    logger.info(f"Manual analysis exported to: {analysis_path}")
                
                # Optionally log the sklearn model artifact
                # mlflow.sklearn.log_model(model, "model")
                logger.info(f"{run_tag} Final Val F1-Macro: {metrics_val['f1_macro']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    import sys
    from loguru import logger
    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")
    
    train_traditional(args.data_path)
