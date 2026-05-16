import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

from src.data_loader import (
    prepare_data, BASIC_TO_ID, ID_TO_BASIC, FINE_TO_ID, ID_TO_FINE,
    FINE_TO_BASIC_TAXONOMY, BASIC_LABELS, FINE_LABELS,
)
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary
from src.utils.threshold_tuning import optimize_thresholds, apply_thresholds
from src.utils.experiment_config import get_config
from src.deep_learning.models import BiLSTM, TextCNN, FastTextDataset, BertDataset
from src.deep_learning.dl_pipeline import build_fasttext_embedding, build_indobert_embedder, predict_probs

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PARAMS = {
    "LR": {"max_iter": 2000, "random_state": 42},
    "NB": {"alpha": 1.0},
    "SVM": {"kernel": "rbf", "random_state": 42},
}

FEATURE_CONFIGS = [
    {"name": "Unigram_BoW", "ngram_range": (1, 1), "vectorizer": "count", "max_features": 5000},
    {"name": "Unigram_TFIDF", "ngram_range": (1, 1), "vectorizer": "tfidf", "max_features": 5000},
    {"name": "Bigram_BoW", "ngram_range": (1, 2), "vectorizer": "count", "max_features": 10000},
    {"name": "Bigram_TFIDF", "ngram_range": (1, 2), "vectorizer": "tfidf", "max_features": 10000},
    {"name": "Trigram_BoW", "ngram_range": (1, 3), "vectorizer": "count", "max_features": 15000},
    {"name": "Trigram_TFIDF", "ngram_range": (1, 3), "vectorizer": "tfidf", "max_features": 15000},
]

TRANSFORMER_MODELS = {
    "IndoBERT": "indobenchmark/indobert-base-p1",
    "IndoBERT-LEM": "indolem/indobert-base-uncased",
    "IndoBERTweet": "indolem/indobertweet-base-uncased",
    "XLM-R": "xlm-roberta-base",
    "mmBERT": "jhu-clsp/mmBERT-base",
}

TRANSFORMER_LRS = [2e-5, 3e-5, 5e-5]
TRANSFORMER_BS = 32


def _make_vectorizer(feat_conf):
    if feat_conf['vectorizer'] == 'count':
        return CountVectorizer(ngram_range=feat_conf['ngram_range'], max_features=feat_conf['max_features'])
    else:
        return TfidfVectorizer(ngram_range=feat_conf['ngram_range'], max_features=feat_conf['max_features'])


def get_seen_labels(y):
    return np.where(y.sum(axis=0) > 0)[0]


class LabelPowersetConverter:
    _NO_LABEL = '__NO_LABEL__'

    def __init__(self, id_to_label):
        self.id_to_label = id_to_label
        self.label_to_id = {v: k for k, v in id_to_label.items()}
        self.class_to_str = {}
        self.str_to_class = {}

    def fit(self, y_binary):
        seen = set()
        class_idx = 0
        for row in y_binary:
            active_labels = tuple(sorted([self.id_to_label[i] for i, val in enumerate(row) if val == 1]))
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
            active_labels = tuple(sorted([self.id_to_label[i] for i, val in enumerate(row) if val == 1]))
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


def evaluate_traditional_test(data_path, config, output_dir):
    exp_tag = config["experiment"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Traditional ML on test set ({exp_tag})")
    logger.info(f"{'='*60}")

    prep_mode = 'traditional'
    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        basic_to_id, id_to_basic,
        fine_to_id, id_to_fine,
        taxonomy,
    ) = prepare_data(data_path, preprocessing_mode=prep_mode)

    train_texts = train_df['preprocessed_text'].tolist()
    val_texts = val_df['preprocessed_text'].tolist()
    test_texts = test_df['preprocessed_text'].tolist()
    test_texts_raw = test_df['text'].tolist()

    scenarios = [
        ("BR_Basic", y_train_basic, y_val_basic, y_test_basic, id_to_basic, False, False),
        ("LP_Basic", y_train_basic, y_val_basic, y_test_basic, id_to_basic, True, False),
        ("BR_Fine", y_train_fine, y_val_fine, y_test_fine, id_to_fine, False, True),
    ]

    use_threshold_tuning = config["threshold_tuning"]
    all_results = []

    for scenario, y_train, y_val, y_test, id_to_class, is_powerset, is_fine in scenarios:
        lp_converter = None
        y_train_lp = None
        if is_powerset:
            lp_converter = LabelPowersetConverter(id_to_class)
            lp_converter.fit(y_train)
            y_train_lp = lp_converter.transform(y_train)

        seen_labels = get_seen_labels(y_train)
        n_seen = len(seen_labels)
        n_total = y_train.shape[1]

        for feat_conf in FEATURE_CONFIGS:
            for model_type in ["LR", "NB", "SVM"]:
                feat_name = feat_conf['name']
                run_tag = f"{scenario}_{feat_name}_{model_type}"

                save_dir = f"{config['saved_models_dir']}/{run_tag}"
                model_joblib = f"{save_dir}/model.joblib"
                vectorizer_joblib = f"{save_dir}/vectorizer.joblib"

                val_probs = None
                thresholds = None
                skip_retrain = False

                if os.path.exists(model_joblib) and os.path.exists(vectorizer_joblib):
                    vectorizer = joblib.load(vectorizer_joblib)
                    n_features_expected = feat_conf['max_features']
                    vocab_size = len(vectorizer.vocabulary_)
                    if vocab_size == n_features_expected:
                        skip_retrain = True
                        logger.info(f"  Loading {run_tag} from saved model (skip retrain)...")
                        model = joblib.load(model_joblib)
                        X_val = vectorizer.transform(val_texts)
                        X_test = vectorizer.transform(test_texts)
                    else:
                        logger.warning(f"  Saved vectorizer has {vocab_size} features, expected {n_features_expected}. Retraining...")
                        os.remove(model_joblib)
                        os.remove(vectorizer_joblib)
                        skip_retrain = False

                if not skip_retrain:
                    logger.info(f"  Retraining {run_tag} on train only...")
                    vectorizer = _make_vectorizer(feat_conf)
                    X_train = vectorizer.fit_transform(train_texts)
                    X_val = vectorizer.transform(val_texts)
                    X_test = vectorizer.transform(test_texts)

                    base_model_cls = {"LR": LogisticRegression, "NB": MultinomialNB, "SVM": SVC}[model_type]

                    if is_powerset:
                        model = base_model_cls(**MODEL_PARAMS[model_type])
                        model.fit(X_train, y_train_lp)
                    else:
                        model = MultiOutputClassifier(base_model_cls(**MODEL_PARAMS[model_type]))
                        if n_seen < n_total:
                            model.fit(X_train, y_train[:, seen_labels])
                        else:
                            model.fit(X_train, y_train)

                    os.makedirs(save_dir, exist_ok=True)
                    joblib.dump(model, model_joblib)
                    joblib.dump(vectorizer, vectorizer_joblib)

                if is_powerset:
                    preds_val_lp = model.predict(X_val)
                    preds_val_bin = lp_converter.inverse_transform(preds_val_lp)
                    preds_test_lp = model.predict(X_test)
                    preds_test_bin = lp_converter.inverse_transform(preds_test_lp)
                else:
                    if n_seen < n_total:
                        preds_val_filtered = model.predict(X_val)
                        preds_val_bin = np.zeros((len(y_val), n_total), dtype=np.int32)
                        preds_val_bin[:, seen_labels] = preds_val_filtered
                        preds_test_filtered = model.predict(X_test)
                        preds_test_bin = np.zeros((len(y_test), n_total), dtype=np.int32)
                        preds_test_bin[:, seen_labels] = preds_test_filtered
                    else:
                        preds_val_bin = model.predict(X_val)
                        preds_test_bin = model.predict(X_test)

                    if use_threshold_tuning and model_type in ("LR", "SVM"):
                        try:
                            proba_val = np.array([est.predict_proba(X_val)[:, 1] for est in model.estimators_]).T
                            full_val_probs = np.zeros((len(y_val), n_total), dtype=np.float64)
                            full_val_probs[:, seen_labels] = proba_val
                            val_probs = full_val_probs
                            thresholds = optimize_thresholds(y_val, val_probs, id_to_class)
                            preds_val_bin = apply_thresholds(val_probs, thresholds)
                            proba_test = np.array([est.predict_proba(X_test)[:, 1] for est in model.estimators_]).T
                            full_test_probs = np.zeros((len(y_test), n_total), dtype=np.float64)
                            full_test_probs[:, seen_labels] = proba_test
                            preds_test_bin = apply_thresholds(full_test_probs, thresholds)
                        except Exception as e:
                            logger.warning(f"Threshold tuning failed for {run_tag}: {e}")

                class_to_parent = taxonomy if is_fine else {}
                val_metrics = compute_all_metrics_binary(y_val, preds_val_bin, id_to_class, class_to_parent)
                test_metrics = compute_all_metrics_binary(y_test, preds_test_bin, id_to_class, class_to_parent)

                result = {
                    "pipeline": "Traditional ML",
                    "run_name": run_tag,
                    "scenario": scenario,
                    "feature": feat_name,
                    "model_type": model_type,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                }
                if thresholds is not None:
                    for i, t in enumerate(thresholds):
                        result[f"threshold_{id_to_class[i]}"] = t

                all_results.append(result)
                logger.info(f"  {run_tag} | Val F1: {val_metrics['f1_macro']:.4f} | Test F1: {test_metrics['f1_macro']:.4f}")

    return all_results


def evaluate_dl_test(data_path, config, output_dir):
    exp_tag = config["experiment"]
    saved_models_dir = config["saved_models_dir"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Deep Learning on test set ({exp_tag})")
    logger.info(f"{'='*60}")

    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        basic_to_id, id_to_basic,
        fine_to_id, id_to_fine,
        taxonomy,
    ) = prepare_data(data_path, preprocessing_mode='deep_learning')

    train_texts = train_df['preprocessed_text'].tolist()
    val_texts = val_df['preprocessed_text'].tolist()
    test_texts = test_df['preprocessed_text'].tolist()
    test_texts_raw = test_df['text'].tolist()

    y_train_fine = y_train_fine.astype(np.float32)
    y_val_fine = y_val_fine.astype(np.float32)
    y_test_fine = y_test_fine.astype(np.float32)
    y_train_basic = y_train_basic.astype(np.float32)
    y_val_basic = y_val_basic.astype(np.float32)
    y_test_basic = y_test_basic.astype(np.float32)

    max_len = config["dl_max_len"]
    use_threshold_tuning = config["threshold_tuning"]
    all_results = []

    dl_configs = [
        ("bilstm", "fasttext", "basic", config["dl_bilstm_hidden"], y_train_basic, y_val_basic, y_test_basic, id_to_basic, {}),
        ("bilstm", "fasttext", "fine", config["dl_bilstm_hidden"], y_train_fine, y_val_fine, y_test_fine, id_to_fine, taxonomy),
        ("cnn", "fasttext", "basic", 100, y_train_basic, y_val_basic, y_test_basic, id_to_basic, {}),
        ("cnn", "fasttext", "fine", 100, y_train_fine, y_val_fine, y_test_fine, id_to_fine, taxonomy),
        ("bilstm", "indobert", "basic", config["dl_bilstm_hidden"], y_train_basic, y_val_basic, y_test_basic, id_to_basic, {}),
        ("bilstm", "indobert", "fine", config["dl_bilstm_hidden"], y_train_fine, y_val_fine, y_test_fine, id_to_fine, taxonomy),
        ("cnn", "indobert", "basic", 100, y_train_basic, y_val_basic, y_test_basic, id_to_basic, {}),
        ("cnn", "indobert", "fine", 100, y_train_fine, y_val_fine, y_test_fine, id_to_fine, taxonomy),
    ]

    for model_type, emb_type, target, param_value, y_train, y_val, y_test, id_to_class, class_to_parent in dl_configs:
        is_basic = (target == 'basic')
        num_classes = len(id_to_class)
        run_tag = f"{model_type}_{emb_type}_{target}_{'hid' if model_type == 'bilstm' else 'filt'}{param_value}"

        save_dir = f"{saved_models_dir}/{run_tag}"
        model_path = f"{save_dir}/model.pt"

        if not os.path.exists(model_path):
            logger.warning(f"Model not found: {model_path}, skipping")
            continue

        logger.info(f"  Loading {run_tag} from {save_dir}...")

        if emb_type == 'fasttext':
            word_to_id, emb_matrix, embed_dim = build_fasttext_embedding(train_texts)
            test_ds = FastTextDataset(test_texts, y_test, word_to_id, max_len=max_len)
            val_ds = FastTextDataset(val_texts, y_val, word_to_id, max_len=max_len)
            bert_mode = False

            if model_type == 'bilstm':
                classifier = BiLSTM(embed_dim, param_value, num_classes, num_layers=config["dl_bilstm_layers"])
            else:
                classifier = TextCNN(embed_dim, num_classes, num_filters=param_value)

            embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix), freeze=True, padding_idx=0)

            class FastTextModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = embedding_layer
                    self.classifier = classifier
                def forward(self, x):
                    return self.classifier(self.embedding(x))

            model = FastTextModel()
        else:
            tokenizer, bert_model, embed_dim = build_indobert_embedder(freeze=True)
            test_ds = BertDataset(test_texts, y_test, tokenizer, max_len=max_len)
            val_ds = BertDataset(val_texts, y_val, tokenizer, max_len=max_len)
            bert_mode = True

            if model_type == 'bilstm':
                classifier = BiLSTM(embed_dim, param_value, num_classes, num_layers=config["dl_bilstm_layers"])
            else:
                classifier = TextCNN(embed_dim, num_classes, num_filters=param_value)

            class IndoBERTModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bert = bert_model
                    self.classifier = classifier
                def forward(self, input_ids, attention_mask):
                    bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    return self.classifier(bert_outputs.last_hidden_state)

            model = IndoBERTModel()

        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        test_loader = DataLoader(test_ds, batch_size=32)
        val_loader = DataLoader(val_ds, batch_size=32)

        val_probs_arr, y_val_true = predict_probs(model, val_loader, DEVICE, bert_mode=bert_mode)
        test_probs_arr, y_test_true = predict_probs(model, test_loader, DEVICE, bert_mode=bert_mode)

        thresholds = None
        threshold_path = f"{save_dir}/thresholds.npy"
        if use_threshold_tuning and os.path.exists(threshold_path):
            thresholds = np.load(threshold_path, allow_pickle=True)
            y_val_pred = apply_thresholds(val_probs_arr, thresholds)
            y_test_pred = apply_thresholds(test_probs_arr, thresholds)
        elif use_threshold_tuning:
            thresholds = optimize_thresholds(y_val_true, val_probs_arr, id_to_class)
            y_val_pred = apply_thresholds(val_probs_arr, thresholds)
            y_test_pred = apply_thresholds(test_probs_arr, thresholds)
        else:
            y_val_pred = (val_probs_arr >= 0.5).astype(np.int32)
            y_test_pred = (test_probs_arr >= 0.5).astype(np.int32)

        val_metrics = compute_all_metrics_binary(y_val_true, y_val_pred, id_to_class, class_to_parent)
        test_metrics = compute_all_metrics_binary(y_test_true, y_test_pred, id_to_class, class_to_parent)

        result = {
            "pipeline": "Deep Learning",
            "run_name": run_tag,
            "model_type": model_type,
            "embedding": emb_type,
            "target_level": target,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        all_results.append(result)
        logger.info(f"  {run_tag} | Val F1: {val_metrics['f1_macro']:.4f} | Test F1: {test_metrics['f1_macro']:.4f}")

    return all_results


def evaluate_transformers_test(data_path, config, output_dir):
    exp_tag = config["experiment"]
    saved_models_dir = config["saved_models_dir"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating Transformers on test set ({exp_tag})")
    logger.info(f"{'='*60}")

    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        basic_to_id, id_to_basic,
        fine_to_id, id_to_fine,
        taxonomy,
    ) = prepare_data(data_path, preprocessing_mode='transformers')

    test_texts = test_df['preprocessed_text'].tolist()
    val_texts = val_df['preprocessed_text'].tolist()
    test_texts_raw = test_df['text'].tolist()

    y_test_basic = y_test_basic.astype(np.float32)
    y_val_basic = y_val_basic.astype(np.float32)
    y_test_fine = y_test_fine.astype(np.float32)
    y_val_fine = y_val_fine.astype(np.float32)

    use_threshold_tuning = config["threshold_tuning"]
    all_results = []

    for model_name, model_id in TRANSFORMER_MODELS.items():
        for target in ["basic", "fine"]:
            is_basic = (target == "basic")
            num_labels = len(basic_to_id) if is_basic else len(fine_to_id)
            id_to_class = id_to_basic if is_basic else id_to_fine
            y_test = y_test_basic if is_basic else y_test_fine
            y_val = y_val_basic if is_basic else y_val_fine
            class_to_parent = {} if is_basic else taxonomy

            for lr in TRANSFORMER_LRS:
                run_tag = f"{model_name}_{target}_lr{lr:.0e}_bs{TRANSFORMER_BS}"
                save_dir = f"{saved_models_dir}/{run_tag}"

                if not os.path.exists(save_dir) or not os.path.exists(f"{save_dir}/config.json"):
                    logger.warning(f"Model not found: {save_dir}, skipping")
                    continue

                logger.info(f"  Loading {run_tag} from {save_dir}...")

                try:
                    tokenizer = AutoTokenizer.from_pretrained(save_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(save_dir)
                    model.to(DEVICE)
                    model.eval()

                    val_enc = tokenizer(val_texts, padding=True, truncation=True, max_length=config["max_len"], return_tensors="pt")
                    test_enc = tokenizer(test_texts, padding=True, truncation=True, max_length=config["max_len"], return_tensors="pt")

                    with torch.no_grad():
                        val_input_ids = val_enc['input_ids'].to(DEVICE)
                        val_attention_mask = val_enc['attention_mask'].to(DEVICE)
                        val_logits = model(input_ids=val_input_ids, attention_mask=val_attention_mask).logits
                        val_probs = torch.sigmoid(val_logits).cpu().numpy()

                        test_input_ids = test_enc['input_ids'].to(DEVICE)
                        test_attention_mask = test_enc['attention_mask'].to(DEVICE)
                        test_logits = model(input_ids=test_input_ids, attention_mask=test_attention_mask).logits
                        test_probs = torch.sigmoid(test_logits).cpu().numpy()

                    threshold_path = f"{save_dir}/thresholds.npy"
                    if use_threshold_tuning and os.path.exists(threshold_path):
                        thresholds = np.load(threshold_path, allow_pickle=True)
                        y_val_pred = apply_thresholds(val_probs, thresholds)
                        y_test_pred = apply_thresholds(test_probs, thresholds)
                    elif use_threshold_tuning:
                        thresholds = optimize_thresholds(y_val, val_probs, id_to_class)
                        y_val_pred = apply_thresholds(val_probs, thresholds)
                        y_test_pred = apply_thresholds(test_probs, thresholds)
                    else:
                        y_val_pred = (val_probs >= 0.5).astype(np.int32)
                        y_test_pred = (test_probs >= 0.5).astype(np.int32)

                    val_metrics = compute_all_metrics_binary(y_val, y_val_pred, id_to_class, class_to_parent)
                    test_metrics = compute_all_metrics_binary(y_test, y_test_pred, id_to_class, class_to_parent)

                    result = {
                        "pipeline": "Transformers",
                        "run_name": run_tag,
                        "model_name": model_name,
                        "target_level": target,
                        "learning_rate": lr,
                        "batch_size": TRANSFORMER_BS,
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        **{f"test_{k}": v for k, v in test_metrics.items()},
                    }
                    all_results.append(result)
                    logger.info(f"  {run_tag} | Val F1: {val_metrics['f1_macro']:.4f} | Test F1: {test_metrics['f1_macro']:.4f}")

                    del model
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"  Failed to evaluate {run_tag}: {e}")
                    continue

    return all_results


def evaluate_all_test(data_path, experiment, output_dir, mlflow_uri):
    config = get_config(experiment)

    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    trad_results = evaluate_traditional_test(data_path, config, output_dir)
    all_results.extend(trad_results)

    dl_results = evaluate_dl_test(data_path, config, output_dir)
    all_results.extend(dl_results)

    tf_results = evaluate_transformers_test(data_path, config, output_dir)
    all_results.extend(tf_results)

    if not all_results:
        logger.error("No results generated!")
        return

    df = pd.DataFrame(all_results)

    output_cols = [
        "run_name",
        "val_f1_micro", "val_f1_macro", "val_f1_weighted", "val_hamming_loss", "val_subset_accuracy",
        "test_f1_micro", "test_f1_macro", "test_f1_weighted", "test_hamming_loss", "test_subset_accuracy",
    ]
    nice_names = {
        "run_name": "Percobaan",
        "val_f1_micro": "Val F1-Micro",
        "val_f1_macro": "Val F1-Macro",
        "val_f1_weighted": "Val F1-Weighted",
        "val_hamming_loss": "Val Hamming Loss",
        "val_subset_accuracy": "Val EMR",
        "test_f1_micro": "Test F1-Micro",
        "test_f1_macro": "Test F1-Macro",
        "test_f1_weighted": "Test F1-Weighted",
        "test_hamming_loss": "Test Hamming Loss",
        "test_subset_accuracy": "Test EMR",
    }

    available = [c for c in output_cols if c in df.columns]
    df_display = df[available].copy()
    df_display = df_display.rename(columns=nice_names)

    for c in df_display.columns:
        if c != "Percobaan":
            df_display[c] = df_display[c].round(4)

    df_display = df_display.sort_values("Percobaan")

    df_display.to_csv(f"{output_dir}/{experiment}_all_val_test.csv", index=False)
    df_display.to_excel(f"{output_dir}/{experiment}_all_val_test.xlsx", index=False, engine='openpyxl')

    logger.info(f"\n{'='*60}")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info(f"  - {experiment}_all_val_test.csv")
    logger.info(f"  - {experiment}_all_val_test.xlsx")
    logger.info(f"  Total: {len(df_display)} runs")
    logger.info(f"{'='*60}")

    return df_display


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/new_all.json")
    parser.add_argument("--experiment", type=str, default="exp1", choices=["exp1", "exp2"])
    parser.add_argument("--output", type=str, default="quantitative_result")
    parser.add_argument("--mlflow_uri", type=str, default="http://localhost:8002")
    args = parser.parse_args()

    evaluate_all_test(args.data_path, args.experiment, args.output, args.mlflow_uri)
