import os
import sys
import argparse
import numpy as np
from loguru import logger
import mlflow

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

from src.data_loader import prepare_data
from src.utils.preprocessing import preprocess_batch
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary


class LabelPowersetConverter:
    def __init__(self, id_to_label):
        self.id_to_label = id_to_label
        self.label_to_id = {v: k for k, v in id_to_label.items()}
        self.class_to_str = {}
        self.str_to_class = {}

    def fit(self, y_binary):
        seen = set()
        class_idx = 0
        for row in y_binary:
            active_labels = tuple(sorted(
                [self.id_to_label[i] for i, val in enumerate(row) if val == 1]
            ))
            if not active_labels:
                active_labels = ('__NO_LABEL__',)
            if active_labels not in seen:
                seen.add(active_labels)
                combo_str = ','.join(active_labels) if active_labels[0] != '__NO_LABEL__' else '__NO_LABEL__'
                self.class_to_str[class_idx] = combo_str
                self.str_to_class[combo_str] = class_idx
                class_idx += 1
        return self

    def transform(self, y_binary):
        result = []
        for row in y_binary:
            active_labels = tuple(sorted(
                [self.id_to_label[i] for i, val in enumerate(row) if val == 1]
            ))
            if not active_labels:
                active_labels = ('__NO_LABEL__',)
            combo_str = ','.join(active_labels) if active_labels[0] != '__NO_LABEL__' else '__NO_LABEL__'
            result.append(self.str_to_class.get(combo_str, -1))
        return np.array(result)

    def inverse_transform(self, y_int):
        n_labels = len(self.label_to_id)
        result = np.zeros((len(y_int), n_labels), dtype=np.int32)
        for i, cls in enumerate(y_int):
            combo_str = self.class_to_str.get(cls, '__NO_LABEL__')
            if combo_str != '__NO_LABEL__':
                for label in combo_str.split(','):
                    if label in self.label_to_id:
                        result[i, self.label_to_id[label]] = 1
        return result


def get_seen_labels(y):
    return np.where(y.sum(axis=0) > 0)[0]


def train_traditional(data_path: str):
    logger.info("Loading data...")
    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    ) = prepare_data(data_path)

    logger.info("Using preprocessed text from data_loader...")
    train_texts = train_df['preprocessed_text'].tolist()
    val_texts = val_df['preprocessed_text'].tolist()
    test_texts = test_df['preprocessed_text'].tolist()
    test_texts_raw = test_df['text'].tolist()

    feature_configs = [
        {"name": "Unigram_BoW", "ngram_range": (1, 1), "vectorizer": "count", "max_features": 5000},
        {"name": "Unigram_TFIDF", "ngram_range": (1, 1), "vectorizer": "tfidf", "max_features": 5000},
        {"name": "Bigram_BoW", "ngram_range": (1, 2), "vectorizer": "count", "max_features": 10000},
        {"name": "Bigram_TFIDF", "ngram_range": (1, 2), "vectorizer": "tfidf", "max_features": 10000},
        {"name": "Trigram_BoW", "ngram_range": (1, 3), "vectorizer": "count", "max_features": 15000},
        {"name": "Trigram_TFIDF", "ngram_range": (1, 3), "vectorizer": "tfidf", "max_features": 15000},
    ]

    models_def = {
        "LR": LogisticRegression(max_iter=2000, random_state=42),
        "NB": MultinomialNB(alpha=1.0),
        "SVM": SVC(kernel='rbf', random_state=42),
    }

    mlflow.set_experiment("Traditional_ML_MultiLabel")

    scenarios = ["BR_Basic", "LP_Basic", "BR_Fine"]
    total_runs = len(scenarios) * len(feature_configs) * len(models_def)
    logger.info(f"Starting {total_runs} total runs across {len(scenarios)} scenarios...")

    for scenario in scenarios:
        logger.info(f"\n=== SCENARIO: {scenario} ===")

        if scenario == "BR_Basic":
            y_train = y_train_basic
            y_val = y_val_basic
            y_test = y_test_basic
            id_to_class = ID_TO_BASIC
            is_powerset = False
            is_fine = False
            taxonomy = {}
        elif scenario == "LP_Basic":
            y_train = y_train_basic
            y_val = y_val_basic
            y_test = y_test_basic
            id_to_class = ID_TO_BASIC
            is_powerset = True
            is_fine = False
            taxonomy = {}
            lp_converter = LabelPowersetConverter(ID_TO_BASIC)
            lp_converter.fit(y_train_basic)
            y_train_lp = lp_converter.transform(y_train_basic)
            logger.info(f"Label Powerset: {len(lp_converter.str_to_class)} unique class combinations")
        else:  # BR_Fine
            y_train = y_train_fine
            y_val = y_val_fine
            y_test = y_test_fine
            id_to_class = ID_TO_FINE
            is_powerset = False
            is_fine = True
            taxonomy = FINE_TO_BASIC_TAXONOMY

        seen_labels = get_seen_labels(y_train)
        n_seen = len(seen_labels)
        n_total = y_train.shape[1]
        logger.info(f"Labels with positive examples: {n_seen}/{n_total}")

        for feat_conf in feature_configs:
            logger.info(f"--- Feature config: {feat_conf['name']} ---")

            if feat_conf['vectorizer'] == 'count':
                vectorizer = CountVectorizer(
                    ngram_range=feat_conf['ngram_range'],
                    max_features=feat_conf['max_features'],
                )
            else:
                vectorizer = TfidfVectorizer(
                    ngram_range=feat_conf['ngram_range'],
                    max_features=feat_conf['max_features'],
                )

            X_train = vectorizer.fit_transform(train_texts)
            X_val = vectorizer.transform(val_texts)
            X_test = vectorizer.transform(test_texts)

            for model_abbr, base_model in models_def.items():
                run_tag = f"{scenario}_{feat_conf['name']}_{model_abbr}"
                logger.info(f"Training {run_tag}...")

                with mlflow.start_run(run_name=run_tag):
                    mlflow.log_param("scenario", scenario)
                    mlflow.log_param("feature_extraction", feat_conf['name'])
                    mlflow.log_param("ngram_range", str(feat_conf['ngram_range']))
                    mlflow.log_param("max_features", feat_conf['max_features'])
                    mlflow.log_param("model_type", model_abbr)

                    for param_name, param_val in base_model.get_params().items():
                        mlflow.log_param(param_name, param_val)

                    if is_powerset:
                        model = base_model
                        model.fit(X_train, y_train_lp)
                        preds_val_lp = model.predict(X_val)
                        preds_test_lp = model.predict(X_test)
                        preds_val_bin = lp_converter.inverse_transform(preds_val_lp)
                        preds_test_bin = lp_converter.inverse_transform(preds_test_lp)
                    else:
                        model = MultiOutputClassifier(base_model)
                        if n_seen < n_total:
                            y_train_filtered = y_train[:, seen_labels]
                            model.fit(X_train, y_train_filtered)
                            preds_val_filtered = model.predict(X_val)
                            preds_test_filtered = model.predict(X_test)
                            preds_val_bin = np.zeros((len(y_val), n_total), dtype=np.int32)
                            preds_test_bin = np.zeros((len(y_test), n_total), dtype=np.int32)
                            preds_val_bin[:, seen_labels] = preds_val_filtered
                            preds_test_bin[:, seen_labels] = preds_test_filtered
                        else:
                            model.fit(X_train, y_train)
                            preds_val_bin = model.predict(X_val)
                            preds_test_bin = model.predict(X_test)

                    metrics_val = compute_all_metrics_binary(
                        y_val,
                        preds_val_bin,
                        id_to_class,
                        taxonomy,
                    )
                    for k, v in metrics_val.items():
                        mlflow.log_metric(f"val_{k}", v)

                    metrics_test = compute_all_metrics_binary(
                        y_test,
                        preds_test_bin,
                        id_to_class,
                        taxonomy,
                    )
                    for k, v in metrics_test.items():
                        mlflow.log_metric(f"test_{k}", v)

                    os.makedirs("analysis", exist_ok=True)

                    analysis_path = f"analysis/manual_analysis_{run_tag}.csv"
                    if is_fine:
                        save_manual_analysis_binary(
                            test_texts_raw,
                            y_test_fine, preds_test_bin,
                            ID_TO_FINE, FINE_TO_BASIC_TAXONOMY,
                            analysis_path,
                        )
                    else:
                        preds_basic_bin = preds_test_bin
                        save_manual_analysis_binary(
                            test_texts_raw,
                            y_test_basic, preds_basic_bin,
                            ID_TO_BASIC, {},
                            analysis_path,
                            y_true_extra=y_test_fine,
                            y_pred_extra=None,
                            id_to_extra=ID_TO_FINE,
                        )
                    logger.info(f"Manual analysis exported to: {analysis_path}")

                    logger.info(
                        f"{run_tag} | Val F1-Macro: {metrics_val['f1_macro']:.4f} | "
                        f"Test F1-Macro: {metrics_test['f1_macro']:.4f}"
                    )

                    try:
                        if is_powerset:
                            mlflow.sklearn.log_model(model, "model")
                        elif n_seen < n_total:
                            wrapper_cls = base_model.__class__
                            wrapper = MultiOutputClassifier(wrapper_cls(**base_model.get_params()))
                            if hasattr(model, 'estimators_'):
                                wrapper.estimators_ = model.estimators_
                            mlflow.sklearn.log_model(wrapper, "model")
                        else:
                            mlflow.sklearn.log_model(model, "model")
                    except Exception as e:
                        logger.warning(f"Failed to log model artifact: {e}")

    logger.info("\n=== Pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")

    train_traditional(args.data_path)