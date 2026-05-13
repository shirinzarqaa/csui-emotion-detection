import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from loguru import logger
import mlflow
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

from src.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary
from src.utils.checkpoint import CheckpointManager


class LabelPowersetConverter:
    def __init__(self, id_to_label):
        self.id_to_label = id_to_label
        self.label_to_id = {v: k for k, v in id_to_label.items()}
        self.class_to_str = {}
        self.str_to_class = {}

    _NO_LABEL = '__NO_LABEL__'

    def fit(self, y_binary):
        seen = set()
        class_idx = 0
        for row in y_binary:
            active_labels = tuple(sorted(
                [self.id_to_label[i] for i, val in enumerate(row) if val == 1]
            ))
            if not active_labels:
                active_labels = (self._NO_LABEL,)
            if active_labels not in seen:
                seen.add(active_labels)
                combo_str = ','.join(active_labels) if active_labels[0] != self._NO_LABEL else self._NO_LABEL
                self.class_to_str[class_idx] = combo_str
                self.str_to_class[combo_str] = class_idx
                class_idx += 1
        if self._NO_LABEL not in self.str_to_class:
            self.class_to_str[class_idx] = self._NO_LABEL
            self.str_to_class[self._NO_LABEL] = class_idx
        return self

    def transform(self, y_binary):
        result = []
        for row in y_binary:
            active_labels = tuple(sorted(
                [self.id_to_label[i] for i, val in enumerate(row) if val == 1]
            ))
            if not active_labels:
                combo_str = self._NO_LABEL
            else:
                combo_str = ','.join(active_labels)
            if combo_str not in self.str_to_class:
                combo_str = self._NO_LABEL
            result.append(self.str_to_class[combo_str])
        return np.array(result)

    def inverse_transform(self, y_int):
        n_labels = len(self.label_to_id)
        result = np.zeros((len(y_int), n_labels), dtype=np.int32)
        for i, cls in enumerate(y_int):
            combo_str = self.class_to_str.get(cls, '')
            if combo_str and combo_str != self._NO_LABEL:
                for label in combo_str.split(','):
                    if label in self.label_to_id:
                        result[i, self.label_to_id[label]] = 1
        return result


def get_seen_labels(y):
    return np.where(y.sum(axis=0) > 0)[0]


def _make_vectorizer(feat_conf):
    if feat_conf['vectorizer'] == 'count':
        return CountVectorizer(
            ngram_range=feat_conf['ngram_range'],
            max_features=feat_conf['max_features'],
        )
    else:
        return TfidfVectorizer(
            ngram_range=feat_conf['ngram_range'],
            max_features=feat_conf['max_features'],
        )


def _make_model(model_type, model_params):
    cls = {"LR": LogisticRegression, "NB": MultinomialNB, "SVM": SVC}[model_type]
    return cls(**model_params)


MODEL_PARAMS = {
    "LR": {"max_iter": 2000, "random_state": 42},
    "NB": {"alpha": 1.0},
    "SVM": {"kernel": "rbf", "random_state": 42},
}


def _run_single_phase1(
    run_tag, scenario, feat_conf, model_type,
    train_texts, val_texts, val_texts_raw,
    y_train, y_val, y_val_basic, y_val_fine,
    id_to_class, taxonomy, is_powerset, is_fine,
    y_train_lp, lp_converter,
    seen_labels, n_seen, n_total,
    ID_TO_BASIC, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY,
):
    result = None
    try:
        mlflow.set_experiment("Traditional_ML_MultiLabel")
        vectorizer = _make_vectorizer(feat_conf)
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        with mlflow.start_run(run_name=run_tag):
            mlflow.set_tag("phase", "experimentation")
            mlflow.log_param("scenario", scenario)
            mlflow.log_param("feature_extraction", feat_conf['name'])
            mlflow.log_param("ngram_range", str(feat_conf['ngram_range']))
            mlflow.log_param("max_features", feat_conf['max_features'])
            mlflow.log_param("model_type", model_type)

            base_model = _make_model(model_type, MODEL_PARAMS[model_type])
            for param_name, param_val in base_model.get_params().items():
                mlflow.log_param(param_name, param_val)

            if is_powerset:
                model = _make_model(model_type, MODEL_PARAMS[model_type])
                model.fit(X_train, y_train_lp)
                preds_val_lp = model.predict(X_val)
                preds_val_bin = lp_converter.inverse_transform(preds_val_lp)
            else:
                model = MultiOutputClassifier(_make_model(model_type, MODEL_PARAMS[model_type]))
                if n_seen < n_total:
                    y_train_filtered = y_train[:, seen_labels]
                    model.fit(X_train, y_train_filtered)
                    preds_val_filtered = model.predict(X_val)
                    preds_val_bin = np.zeros((len(y_val), n_total), dtype=np.int32)
                    preds_val_bin[:, seen_labels] = preds_val_filtered
                else:
                    model.fit(X_train, y_train)
                    preds_val_bin = model.predict(X_val)

            metrics_val = compute_all_metrics_binary(y_val, preds_val_bin, id_to_class, taxonomy)
            for k, v in metrics_val.items():
                mlflow.log_metric(f"val_{k}", v)

            os.makedirs("analysis", exist_ok=True)
            val_analysis_path = f"analysis/val_analysis_{run_tag}.csv"
            if is_fine:
                save_manual_analysis_binary(
                    val_texts_raw, y_val_fine, preds_val_bin,
                    ID_TO_FINE, FINE_TO_BASIC_TAXONOMY,
                    val_analysis_path,
                )
            else:
                save_manual_analysis_binary(
                    val_texts_raw, y_val_basic, preds_val_bin,
                    ID_TO_BASIC, {}, val_analysis_path,
                    y_true_extra=y_val_fine,
                    y_pred_extra=None,
                    id_to_extra=ID_TO_FINE,
                )

            logger.info(f"{run_tag} | Val F1-Macro: {metrics_val['f1_macro']:.4f}")

            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                logger.warning(f"Failed to log model artifact: {e}")

            result = {
                'scenario': scenario,
                'feature_name': feat_conf['name'],
                'ngram_range': list(feat_conf['ngram_range']),
                'max_features': feat_conf['max_features'],
                'vectorizer_type': feat_conf['vectorizer'],
                'model_type': model_type,
                'val_f1_macro': float(metrics_val['f1_macro']),
                'val_f1_micro': float(metrics_val['f1_micro']),
                'val_subset_accuracy': float(metrics_val['subset_accuracy']),
                'val_hamming_loss': float(metrics_val['hamming_loss']),
            }
    except Exception as e:
        logger.error(f"Run {run_tag} FAILED: {e}")
    return run_tag, result


def train_traditional(data_path: str, n_jobs=4):
    logger.info("Loading data...")
    ckpt = CheckpointManager("checkpoints/traditional_checkpoint.json")

    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    ) = prepare_data(data_path, preprocessing_mode='traditional')

    logger.info("Using preprocessed text from data_loader...")
    train_texts = train_df['preprocessed_text'].tolist()
    val_texts = val_df['preprocessed_text'].tolist()
    test_texts = test_df['preprocessed_text'].tolist()
    val_texts_raw = val_df['text'].tolist()
    test_texts_raw = test_df['text'].tolist()

    feature_configs = [
        {"name": "Unigram_BoW", "ngram_range": (1, 1), "vectorizer": "count", "max_features": 5000},
        {"name": "Unigram_TFIDF", "ngram_range": (1, 1), "vectorizer": "tfidf", "max_features": 5000},
        {"name": "Bigram_BoW", "ngram_range": (1, 2), "vectorizer": "count", "max_features": 10000},
        {"name": "Bigram_TFIDF", "ngram_range": (1, 2), "vectorizer": "tfidf", "max_features": 10000},
        {"name": "Trigram_BoW", "ngram_range": (1, 3), "vectorizer": "count", "max_features": 15000},
        {"name": "Trigram_TFIDF", "ngram_range": (1, 3), "vectorizer": "tfidf", "max_features": 15000},
    ]

    scenarios = ["BR_Basic", "LP_Basic", "BR_Fine"]

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: EXPERIMENTATION — val metrics only, NO test
    # ═══════════════════════════════════════════════════════════════
    mlflow.set_experiment("Traditional_ML_MultiLabel")

    total_runs = len(scenarios) * len(feature_configs) * len(MODEL_PARAMS)
    logger.info(f"PHASE 1: {total_runs} experimentation runs (val metrics only)...")

    run_args = []
    for scenario in scenarios:
        if scenario == "BR_Basic":
            y_train = y_train_basic
            y_val = y_val_basic
            id_to_class = ID_TO_BASIC
            is_powerset = False
            is_fine = False
            taxonomy = {}
            lp_converter = None
            y_train_lp = None
        elif scenario == "LP_Basic":
            y_train = y_train_basic
            y_val = y_val_basic
            id_to_class = ID_TO_BASIC
            is_powerset = True
            is_fine = False
            taxonomy = {}
            lp_converter = LabelPowersetConverter(ID_TO_BASIC)
            lp_converter.fit(y_train_basic)
            y_train_lp = lp_converter.transform(y_train_basic)
            logger.info(f"Label Powerset: {len(lp_converter.str_to_class)} unique class combinations")
        else:
            y_train = y_train_fine
            y_val = y_val_fine
            id_to_class = ID_TO_FINE
            is_powerset = False
            is_fine = True
            taxonomy = FINE_TO_BASIC_TAXONOMY
            lp_converter = None
            y_train_lp = None

        seen_labels = get_seen_labels(y_train)
        n_seen = len(seen_labels)
        n_total = y_train.shape[1]
        logger.info(f"Labels with positive examples: {n_seen}/{n_total}")

        for feat_conf in feature_configs:
            for model_type in MODEL_PARAMS:
                run_tag = f"{scenario}_{feat_conf['name']}_{model_type}"

                if ckpt.is_completed(run_tag):
                    logger.info(f"Skipping {run_tag} (already completed)")
                    continue

                run_args.append(dict(
                    run_tag=run_tag, scenario=scenario,
                    feat_conf=feat_conf, model_type=model_type,
                    train_texts=train_texts, val_texts=val_texts,
                    val_texts_raw=val_texts_raw,
                    y_train=y_train, y_val=y_val,
                    y_val_basic=y_val_basic, y_val_fine=y_val_fine,
                    id_to_class=id_to_class, taxonomy=taxonomy,
                    is_powerset=is_powerset, is_fine=is_fine,
                    y_train_lp=y_train_lp, lp_converter=lp_converter,
                    seen_labels=seen_labels, n_seen=n_seen, n_total=n_total,
                    ID_TO_BASIC=ID_TO_BASIC, ID_TO_FINE=ID_TO_FINE,
                    FINE_TO_BASIC_TAXONOMY=FINE_TO_BASIC_TAXONOMY,
                ))

    if run_args:
        logger.info(f"Running {len(run_args)} Phase 1 runs in parallel (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_run_single_phase1)(**kwargs) for kwargs in run_args
        )
        for run_tag, result in results:
            if result is not None:
                ckpt.add_phase1_result(result)
                ckpt.mark_completed(run_tag)

    ckpt.mark_phase_complete(1)
    logger.info(f"\nPHASE 1 complete. {len(ckpt.get_phase1_results())} runs logged (val metrics only).")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: FINAL REPORT — retrain best on train+val, test ONCE
    # ═══════════════════════════════════════════════════════════════
    phase1_results = ckpt.get_phase1_results()
    if not phase1_results:
        logger.error("No Phase 1 results found. Cannot proceed to Phase 2.")
        return
    mlflow.set_experiment("Traditional_ML_Final_Test")
    logger.info("\nPHASE 2: Retraining best models on train+val, evaluating on test (ONCE per scenario)...")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_texts = train_val_df['preprocessed_text'].tolist()
    y_train_val_basic = np.concatenate([y_train_basic, y_val_basic], axis=0)
    y_train_val_fine = np.concatenate([y_train_fine, y_val_fine], axis=0)

    for scenario in scenarios:
        scenario_results = [r for r in phase1_results if r['scenario'] == scenario]
        best = max(scenario_results, key=lambda x: x['val_f1_macro'])

        logger.info(f"Best {scenario}: {best['feature_name']} + {best['model_type']} "
                     f"(val F1-Macro: {best['val_f1_macro']:.4f})")

        if scenario == "BR_Basic":
            y_train_val = y_train_val_basic
            y_test = y_test_basic
            id_to_class = ID_TO_BASIC
            is_powerset = False
            is_fine = False
            taxonomy = {}
        elif scenario == "LP_Basic":
            y_train_val = y_train_val_basic
            y_test = y_test_basic
            id_to_class = ID_TO_BASIC
            is_powerset = True
            is_fine = False
            taxonomy = {}
            lp_converter_final = LabelPowersetConverter(ID_TO_BASIC)
            lp_converter_final.fit(y_train_val_basic)
            y_train_val_lp = lp_converter_final.transform(y_train_val_basic)
        else:
            y_train_val = y_train_val_fine
            y_test = y_test_fine
            id_to_class = ID_TO_FINE
            is_powerset = False
            is_fine = True
            taxonomy = FINE_TO_BASIC_TAXONOMY

        feat_conf_best = {
            'name': best['feature_name'],
            'ngram_range': tuple(best['ngram_range']) if isinstance(best['ngram_range'], list) else best['ngram_range'],
            'max_features': best['max_features'],
            'vectorizer': best['vectorizer_type'],
        }
        vectorizer = _make_vectorizer(feat_conf_best)
        X_train_val = vectorizer.fit_transform(train_val_texts)
        X_test = vectorizer.transform(test_texts)

        seen_labels_tv = get_seen_labels(y_train_val)
        n_seen_tv = len(seen_labels_tv)
        n_total_tv = y_train_val.shape[1]

        run_tag = f"FINAL_{scenario}_{best['feature_name']}_{best['model_type']}"

        if ckpt.is_completed(run_tag):
            logger.info(f"Skipping {run_tag} (already completed)")
            continue

        with mlflow.start_run(run_name=run_tag):
            mlflow.set_tag("phase", "final_test")
            mlflow.log_params({
                "scenario": scenario,
                "feature_extraction": best['feature_name'],
                "ngram_range": str(best['ngram_range']),
                "max_features": best['max_features'],
                "model_type": best['model_type'],
                "selected_by_val_f1_macro": best['val_f1_macro'],
            })

            base_model = _make_model(best['model_type'], MODEL_PARAMS[best['model_type']])
            for param_name, param_val in base_model.get_params().items():
                mlflow.log_param(param_name, param_val)

            if is_powerset:
                model = _make_model(best['model_type'], MODEL_PARAMS[best['model_type']])
                model.fit(X_train_val, y_train_val_lp)
                preds_test_lp = model.predict(X_test)
                preds_test_bin = lp_converter_final.inverse_transform(preds_test_lp)
            else:
                model = MultiOutputClassifier(_make_model(best['model_type'], MODEL_PARAMS[best['model_type']]))
                if n_seen_tv < n_total_tv:
                    y_train_val_filtered = y_train_val[:, seen_labels_tv]
                    model.fit(X_train_val, y_train_val_filtered)
                    preds_test_filtered = model.predict(X_test)
                    preds_test_bin = np.zeros((len(y_test), n_total_tv), dtype=np.int32)
                    preds_test_bin[:, seen_labels_tv] = preds_test_filtered
                else:
                    model.fit(X_train_val, y_train_val)
                    preds_test_bin = model.predict(X_test)

            metrics_test = compute_all_metrics_binary(y_test, preds_test_bin, id_to_class, taxonomy)
            for k, v in metrics_test.items():
                mlflow.log_metric(f"test_{k}", v)

            os.makedirs("analysis", exist_ok=True)
            analysis_path = f"analysis/final_test_analysis_{scenario}.csv"
            if is_fine:
                save_manual_analysis_binary(
                    test_texts_raw, y_test_fine, preds_test_bin,
                    ID_TO_FINE, FINE_TO_BASIC_TAXONOMY, analysis_path,
                )
            else:
                save_manual_analysis_binary(
                    test_texts_raw, y_test_basic, preds_test_bin,
                    ID_TO_BASIC, {}, analysis_path,
                    y_true_extra=y_test_fine,
                    y_pred_extra=None,
                    id_to_extra=ID_TO_FINE,
                )

            logger.info(f"FINAL {scenario} | Test F1-Macro: {metrics_test['f1_macro']:.4f} | "
                        f"Subset Acc: {metrics_test['subset_accuracy']:.4f} | "
                        f"Hamming Loss: {metrics_test['hamming_loss']:.4f}")

            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                logger.warning(f"Failed to log model artifact: {e}")

            ckpt.mark_completed(run_tag)

    ckpt.mark_phase_complete(2)
    logger.info("\n=== Pipeline complete ===")
    logger.info("Phase 1: 54 runs with val metrics → analysis/val_analysis_*.csv")
    logger.info("Phase 2: 3 final runs with test metrics → analysis/final_test_analysis_*.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    logger.add(sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO")

    train_traditional(args.data_path, n_jobs=args.n_jobs)
