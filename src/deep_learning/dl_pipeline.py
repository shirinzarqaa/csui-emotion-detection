import argparse
import os
import sys
import gzip
import urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
from loguru import logger

from src.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary
from src.deep_learning.models import BiLSTM, TextCNN, FastTextDataset, BertDataset
from src.utils.checkpoint import CheckpointManager


CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.fasttext_cache')
FASTTEXT_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _download_fasttext():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, 'cc.id.300.vec.gz')
    if not os.path.exists(cache_path):
        logger.info(f"Downloading FastText vectors from {FASTTEXT_URL} ...")
        try:
            urllib.request.urlretrieve(FASTTEXT_URL, cache_path)
            logger.info(f"FastText vectors cached at {cache_path}")
        except Exception as e:
            logger.error(f"FastText download failed: {e}")
            raise RuntimeError(f"Failed to download FastText vectors: {e}")
    return cache_path


def build_fasttext_embedding(texts, force_reload=False):
    cache_path = _download_fasttext()

    vocab_words = set()
    for t in texts:
        for w in str(t).split():
            vocab_words.add(w)

    logger.info(f"Building FastText embedding for {len(vocab_words)} unique words ...")

    word_to_vec = {}
    embed_dim = None
    with gzip.open(cache_path, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split()
        embed_dim = int(header[1])
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            if word in vocab_words:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                word_to_vec[word] = vec

    word_to_id = {'<PAD>': 0, '<UNK>': 1}
    for i, w in enumerate(sorted(vocab_words), start=2):
        word_to_id[w] = i

    embedding_matrix = np.zeros((len(word_to_id), embed_dim), dtype=np.float32)
    found = 0
    for word, idx in word_to_id.items():
        if word in word_to_vec:
            embedding_matrix[idx] = word_to_vec[word]
            found += 1
        elif word not in ('<PAD>', '<UNK>'):
            embedding_matrix[idx] = np.random.normal(scale=0.02, size=(embed_dim,)).astype(np.float32)

    logger.info(f"FastText coverage: {found}/{len(word_to_id)} words ({100 * found / len(word_to_id):.1f}%)")
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
    return word_to_id, embedding_tensor, embed_dim


def build_indobert_embedder(model_name='indolem/indobert-base-uncased', freeze=True):
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError as e:
        logger.error("transformers library not installed. Install with: pip install transformers")
        raise

    logger.info(f"Loading IndoBERT: {model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        raise

    if freeze:
        for param in bert_model.parameters():
            param.requires_grad = False
        logger.info("IndoBERT weights frozen (feature extractor mode).")

    bert_model.to(DEVICE)
    embed_dim = bert_model.config.hidden_size
    return tokenizer, bert_model, embed_dim


def fine_to_basic_predictions(y_pred_fine, basic_to_id, id_to_fine, fine_to_basic):
    y_pred_basic = np.zeros((len(y_pred_fine), len(basic_to_id)), dtype=np.int32)
    for i in range(len(y_pred_fine)):
        for j in range(len(id_to_fine)):
            if y_pred_fine[i, j] == 1:
                fine_label = id_to_fine[j]
                basic_label = fine_to_basic.get(fine_label)
                if basic_label and basic_label in basic_to_id:
                    y_pred_basic[i, basic_to_id[basic_label]] = 1
    return y_pred_basic


def train_epoch(model, loader, optimizer, criterion, device, bert_mode=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if bert_mode:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
        else:
            token_ids, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(token_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, bert_mode=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        if bert_mode:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask)
        else:
            token_ids, labels = [b.to(device) for b in batch]
            outputs = model(token_ids)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(np.int32)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy().astype(np.int32))
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return total_loss / len(loader), all_labels, all_preds


def _build_model(embedding_type, model_type, param_value, num_classes,
                 word_to_id=None, emb_matrix=None, embed_dim=None,
                 bert_model=None, tokenizer=None):
    if model_type == 'bilstm':
        classifier = BiLSTM(embed_dim, hidden_dim=param_value, num_classes=num_classes)
    elif model_type == 'cnn':
        classifier = TextCNN(embed_dim, num_classes=num_classes, num_filters=param_value)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if embedding_type == 'fasttext':
        embedding_layer = nn.Embedding.from_pretrained(emb_matrix, freeze=True, padding_idx=0)
        embedding_layer.to(DEVICE)

        class FastTextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = embedding_layer
                self.classifier = classifier

            def forward(self, x):
                return self.classifier(self.embedding(x))

        model = FastTextModel()
        bert_mode = False
    else:
        class IndoBERTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = bert_model
                self.classifier = classifier

            def forward(self, input_ids, attention_mask):
                bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                return self.classifier(bert_outputs.last_hidden_state)

        model = IndoBERTModel()
        bert_mode = True

    model.to(DEVICE)
    return model, bert_mode


def _build_datasets(embedding_type, texts, labels, word_to_id=None,
                    tokenizer=None, max_len=128):
    if embedding_type == 'fasttext':
        return FastTextDataset(texts, labels, word_to_id, max_len=max_len)
    else:
        return BertDataset(texts, labels, tokenizer, max_len=max_len)


def train_and_evaluate_val(
    *,
    embedding_type,
    model_type,
    param_value,
    target_level,
    train_texts, val_texts,
    y_train, y_val,
    y_val_basic, y_train_basic,
    BASIC_TO_ID, ID_TO_BASIC,
    FINE_TO_ID, ID_TO_FINE,
    FINE_TO_BASIC_TAXONOMY,
    num_epochs=30,
    batch_size=32,
    lr=1e-3,
    patience=3,
    max_len=128,
    val_texts_raw=None,
):
    is_basic = (target_level == 'basic')
    num_classes = len(BASIC_TO_ID) if is_basic else len(FINE_TO_ID)
    id_to_class = ID_TO_BASIC if is_basic else ID_TO_FINE
    class_to_parent = {} if is_basic else FINE_TO_BASIC_TAXONOMY

    run_tag = f"{model_type}_{embedding_type}_{target_level}_{'hid' if model_type == 'bilstm' else 'filt'}{param_value}"
    logger.info(f"Phase 1 run: {run_tag}")

    if embedding_type == 'fasttext':
        word_to_id, emb_matrix, embed_dim = build_fasttext_embedding(train_texts)
        train_ds = FastTextDataset(train_texts, y_train, word_to_id, max_len=max_len)
        val_ds = FastTextDataset(val_texts, y_val, word_to_id, max_len=max_len)
        bert_mode = False
    else:
        tokenizer, bert_model, embed_dim = build_indobert_embedder(freeze=True)
        train_ds = BertDataset(train_texts, y_train, tokenizer, max_len=max_len)
        val_ds = BertDataset(val_texts, y_val, tokenizer, max_len=max_len)
        bert_mode = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model, bert_mode = _build_model(
        embedding_type, model_type, param_value, num_classes,
        word_to_id=word_to_id if embedding_type == 'fasttext' else None,
        emb_matrix=emb_matrix if embedding_type == 'fasttext' else None,
        embed_dim=embed_dim,
        bert_model=bert_model if embedding_type == 'indobert' else None,
        tokenizer=tokenizer if embedding_type == 'indobert' else None,
    )

    pos_count = y_train.sum(axis=0)
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor(neg_count / (pos_count + 1e-6), dtype=torch.float).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, bert_mode=bert_mode)
        val_loss, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, DEVICE, bert_mode=bert_mode)

        val_metrics = compute_all_metrics_binary(y_val_true, y_val_pred, id_to_class, class_to_parent)
        val_f1 = val_metrics['f1_macro']

        if is_basic:
            logger.info(
                f"Epoch {epoch:2d}/{num_epochs} | "
                f"tr_loss={train_loss:.4f} | vl_loss={val_loss:.4f} | "
                f"vl_f1_m={val_f1:.4f}"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }, step=epoch)
        else:
            y_val_pred_basic = fine_to_basic_predictions(y_val_pred, BASIC_TO_ID, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY)
            val_basic_metrics = compute_all_metrics_binary(y_val_basic, y_val_pred_basic, ID_TO_BASIC, {})

            logger.info(
                f"Epoch {epoch:2d}/{num_epochs} | "
                f"tr_loss={train_loss:.4f} | vl_loss={val_loss:.4f} | "
                f"vl_f1_m={val_f1:.4f} | "
                f"vl_basic_f1_m={val_basic_metrics['f1_macro']:.4f}"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"val_basic_{k}": v for k, v in val_basic_metrics.items()},
            }, step=epoch)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            logger.info(f"  New best val_f1_macro = {best_val_f1:.4f} at epoch {best_epoch}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, DEVICE, bert_mode=bert_mode)
    final_val_metrics = compute_all_metrics_binary(y_val_true, y_val_pred, id_to_class, class_to_parent)
    for k, v in final_val_metrics.items():
        mlflow.log_metric(f"val_{k}", v)
    if not is_basic:
        y_val_pred_basic = fine_to_basic_predictions(y_val_pred, BASIC_TO_ID, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY)
        final_val_basic = compute_all_metrics_binary(y_val_basic, y_val_pred_basic, ID_TO_BASIC, {})
        for k, v in final_val_basic.items():
            mlflow.log_metric(f"val_basic_{k}", v)

    os.makedirs("analysis", exist_ok=True)
    val_analysis_path = f"analysis/val_analysis_{run_tag}.csv"
    if is_basic:
        save_manual_analysis_binary(
            val_texts_raw, y_val_true, y_val_pred,
            ID_TO_BASIC, {}, val_analysis_path,
            y_true_extra=y_val_basic if not is_basic else None,
            y_pred_extra=None,
            id_to_extra=ID_TO_FINE if not is_basic else None,
        )
    else:
        save_manual_analysis_binary(
            val_texts_raw, y_val_true, y_val_pred,
            ID_TO_FINE, FINE_TO_BASIC_TAXONOMY, val_analysis_path,
        )

    try:
        mlflow.pytorch.log_model(model, "model")
        logger.info("Model artifact logged to MLflow.")
    except Exception as e:
        logger.warning(f"Failed to log model artifact: {e}")

    logger.info(f"Phase 1 run {run_tag} finished. Best val_f1_macro={best_val_f1:.4f} at epoch {best_epoch}")

    return {
        'run_tag': run_tag,
        'embedding_type': embedding_type,
        'model_type': model_type,
        'target_level': target_level,
        'val_f1_macro': best_val_f1,
        'best_epoch': best_epoch,
    }


def retrain_and_test(
    *,
    embedding_type,
    model_type,
    param_value,
    target_level,
    best_epoch,
    train_val_texts, test_texts,
    y_train_val, y_test,
    y_test_basic,
    y_train_val_basic,
    BASIC_TO_ID, ID_TO_BASIC,
    FINE_TO_ID, ID_TO_FINE,
    FINE_TO_BASIC_TAXONOMY,
    test_texts_raw,
    batch_size=32,
    lr=1e-3,
    max_len=128,
):
    is_basic = (target_level == 'basic')
    num_classes = len(BASIC_TO_ID) if is_basic else len(FINE_TO_ID)
    id_to_class = ID_TO_BASIC if is_basic else ID_TO_FINE
    class_to_parent = {} if is_basic else FINE_TO_BASIC_TAXONOMY

    run_tag = f"FINAL_{model_type}_{embedding_type}_{target_level}"
    logger.info(f"Phase 2: {run_tag} — retraining on train+val for {best_epoch} epochs, then test ONCE")

    if embedding_type == 'fasttext':
        word_to_id, emb_matrix, embed_dim = build_fasttext_embedding(train_val_texts)
        train_val_ds = FastTextDataset(train_val_texts, y_train_val, word_to_id, max_len=max_len)
        test_ds = FastTextDataset(test_texts, y_test, word_to_id, max_len=max_len)
        bert_mode = False
    else:
        tokenizer, bert_model, embed_dim = build_indobert_embedder(freeze=True)
        train_val_ds = BertDataset(train_val_texts, y_train_val, tokenizer, max_len=max_len)
        test_ds = BertDataset(test_texts, y_test, tokenizer, max_len=max_len)
        bert_mode = True

    train_val_loader = DataLoader(train_val_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model, bert_mode = _build_model(
        embedding_type, model_type, param_value, num_classes,
        word_to_id=word_to_id if embedding_type == 'fasttext' else None,
        emb_matrix=emb_matrix if embedding_type == 'fasttext' else None,
        embed_dim=embed_dim,
        bert_model=bert_model if embedding_type == 'indobert' else None,
        tokenizer=tokenizer if embedding_type == 'indobert' else None,
    )

    pos_count_tv = y_train_val.sum(axis=0)
    neg_count_tv = len(y_train_val) - pos_count_tv
    pos_weight_tv = torch.tensor(neg_count_tv / (pos_count_tv + 1e-6), dtype=torch.float).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tv)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, best_epoch + 1):
        train_loss = train_epoch(model, train_val_loader, optimizer, criterion, DEVICE, bert_mode=bert_mode)
        logger.info(f"  Phase 2 Epoch {epoch}/{best_epoch} | train_loss={train_loss:.4f}")

    _, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, DEVICE, bert_mode=bert_mode)
    test_metrics = compute_all_metrics_binary(y_test_true, y_test_pred, id_to_class, class_to_parent)

    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)

    if not is_basic:
        y_test_pred_basic = fine_to_basic_predictions(y_test_pred, BASIC_TO_ID, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY)
        test_basic = compute_all_metrics_binary(y_test_basic, y_test_pred_basic, ID_TO_BASIC, {})
        for k, v in test_basic.items():
            mlflow.log_metric(f"test_basic_{k}", v)

    os.makedirs("analysis", exist_ok=True)
    analysis_path = f"analysis/final_test_analysis_{run_tag}.csv"
    if is_basic:
        save_manual_analysis_binary(
            test_texts_raw, y_test_true, y_test_pred,
            ID_TO_BASIC, {}, analysis_path,
            y_true_extra=y_test_basic if not is_basic else None,
            y_pred_extra=None,
            id_to_extra=ID_TO_FINE if not is_basic else None,
        )
    else:
        save_manual_analysis_binary(
            test_texts_raw, y_test_true, y_test_pred,
            ID_TO_FINE, FINE_TO_BASIC_TAXONOMY, analysis_path,
        )

    logger.info(f"FINAL {run_tag} | Test F1-Macro: {test_metrics['f1_macro']:.4f} | "
                f"Subset Acc: {test_metrics['subset_accuracy']:.4f} | "
                f"Hamming Loss: {test_metrics['hamming_loss']:.4f}")

    try:
        mlflow.pytorch.log_model(model, "model")
    except Exception as e:
        logger.warning(f"Failed to log model artifact: {e}")

    return test_metrics


def train_dl(data_path):
    logger.info("=== Deep Learning Multi-Label Emotion Classification ===")
    logger.info(f"Device: {DEVICE}")
    ckpt = CheckpointManager("checkpoints/dl_checkpoint.json")

    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    ) = prepare_data(data_path, preprocessing_mode='deep_learning')

    logger.info("Using preprocessed text from data_loader...")
    train_texts = train_df['preprocessed_text'].tolist()
    val_texts = val_df['preprocessed_text'].tolist()
    test_texts = test_df['preprocessed_text'].tolist()
    val_texts_raw = val_df['text'].tolist()
    test_texts_raw = test_df['text'].tolist()
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    y_train_fine = y_train_fine.astype(np.float32)
    y_val_fine = y_val_fine.astype(np.float32)
    y_test_fine = y_test_fine.astype(np.float32)
    y_train_basic = y_train_basic.astype(np.float32)
    y_val_basic = y_val_basic.astype(np.float32)
    y_test_basic = y_test_basic.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: EXPERIMENTATION — val metrics only, NO test
    # ═══════════════════════════════════════════════════════════════
    mlflow.set_experiment("Deep_Learning_MultiLabel")
    phase1_results = ckpt.get_phase1_results()
    logger.info("PHASE 1: Experimentation runs (val metrics only)...")

    embedding_configs = ['fasttext', 'indobert']
    target_levels = ['basic', 'fine']

    common_kwargs = dict(
        train_texts=train_texts,
        val_texts=val_texts,
        y_val_basic=y_val_basic,
        y_train_basic=y_train_basic,
        BASIC_TO_ID=BASIC_TO_ID,
        ID_TO_BASIC=ID_TO_BASIC,
        FINE_TO_ID=FINE_TO_ID,
        ID_TO_FINE=ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY=FINE_TO_BASIC_TAXONOMY,
        val_texts_raw=val_texts_raw,
    )

    for emb_type in embedding_configs:
        for target in target_levels:
            is_basic = (target == 'basic')
            y_train = y_train_basic if is_basic else y_train_fine
            y_val = y_val_basic if is_basic else y_val_fine

            run_tag = f"bilstm_{emb_type}_{target}"

            if ckpt.is_completed(run_tag):
                logger.info(f"Skipping {run_tag} (already completed)")
                continue

            with mlflow.start_run(run_name=run_tag):
                mlflow.set_tag("phase", "experimentation")
                mlflow.log_params({
                    "model_type": "bilstm",
                    "embedding_type": emb_type,
                    "embed_dim": 300 if emb_type == 'fasttext' else 768,
                    "hidden_dim": 128,
                    "target_level": target,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                })
                try:
                    result = train_and_evaluate_val(
                        embedding_type=emb_type,
                        model_type='bilstm',
                        param_value=128,
                        target_level=target,
                        y_train=y_train,
                        y_val=y_val,
                        **common_kwargs,
                    )
                    phase1_results.append(result)
                    ckpt.add_phase1_result(result)
                    ckpt.mark_completed(run_tag)
                except Exception as e:
                    logger.error(f"Run {run_tag} FAILED: {e}")
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))

            run_tag = f"cnn_{emb_type}_{target}"

            if ckpt.is_completed(run_tag):
                logger.info(f"Skipping {run_tag} (already completed)")
                continue

            with mlflow.start_run(run_name=run_tag):
                mlflow.set_tag("phase", "experimentation")
                mlflow.log_params({
                    "model_type": "cnn",
                    "embedding_type": emb_type,
                    "embed_dim": 300 if emb_type == 'fasttext' else 768,
                    "num_filters": 100,
                    "target_level": target,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                })
                try:
                    result = train_and_evaluate_val(
                        embedding_type=emb_type,
                        model_type='cnn',
                        param_value=100,
                        target_level=target,
                        y_train=y_train,
                        y_val=y_val,
                        **common_kwargs,
                    )
                    phase1_results.append(result)
                    ckpt.add_phase1_result(result)
                    ckpt.mark_completed(run_tag)
                except Exception as e:
                    logger.error(f"Run {run_tag} FAILED: {e}")
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))

    logger.info(f"\nPHASE 1 complete. {len(phase1_results)} runs logged (val metrics only).")
    ckpt.mark_phase_complete(1)

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: FINAL REPORT — retrain best on train+val, test ONCE
    # ═══════════════════════════════════════════════════════════════
    phase1_results = ckpt.get_phase1_results()
    if not phase1_results:
        logger.error("No Phase 1 results found. Cannot proceed to Phase 2.")
        return
    mlflow.set_experiment("Deep_Learning_Final_Test")
    logger.info("\nPHASE 2: Retraining best models on train+val, evaluating on test (ONCE per target)...")

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_texts = train_val_df['preprocessed_text'].tolist()
    test_texts = test_df['preprocessed_text'].tolist()
    test_texts_raw = test_df['text'].tolist()

    y_train_val_basic = np.concatenate([y_train_basic, y_val_basic], axis=0)
    y_train_val_fine = np.concatenate([y_train_fine, y_val_fine], axis=0)

    for target in target_levels:
        target_results = [r for r in phase1_results if r['target_level'] == target]
        if not target_results:
            logger.warning(f"No Phase 1 results for target={target}, skipping Phase 2.")
            continue
        best = max(target_results, key=lambda x: x['val_f1_macro'])
        is_basic = (target == 'basic')
        y_train_val = y_train_val_basic if is_basic else y_train_val_fine
        y_test = y_test_basic if is_basic else y_test_fine

        run_tag = f"FINAL_{best['model_type']}_{best['embedding_type']}_{target}"

        if ckpt.is_completed(run_tag):
            logger.info(f"Skipping {run_tag} (already completed)")
            continue

        with mlflow.start_run(run_name=run_tag):
            mlflow.set_tag("phase", "final_test")
            mlflow.log_params({
                "model_type": best['model_type'],
                "embedding_type": best['embedding_type'],
                "target_level": target,
                "best_epoch": best['best_epoch'],
                "selected_by_val_f1_macro": best['val_f1_macro'],
                "hidden_dim" if best['model_type'] == 'bilstm' else "num_filters": 128 if best['model_type'] == 'bilstm' else 100,
            })

            try:
                retrain_and_test(
                    embedding_type=best['embedding_type'],
                    model_type=best['model_type'],
                    param_value=128 if best['model_type'] == 'bilstm' else 100,
                    target_level=target,
                    best_epoch=best['best_epoch'],
                    train_val_texts=train_val_texts,
                    test_texts=test_texts,
                    y_train_val=y_train_val,
                    y_test=y_test,
                    y_test_basic=y_test_basic,
                    y_train_val_basic=y_train_val_basic,
                    BASIC_TO_ID=BASIC_TO_ID,
                    ID_TO_BASIC=ID_TO_BASIC,
                    FINE_TO_ID=FINE_TO_ID,
                    ID_TO_FINE=ID_TO_FINE,
                    FINE_TO_BASIC_TAXONOMY=FINE_TO_BASIC_TAXONOMY,
                    test_texts_raw=test_texts_raw,
                )
                ckpt.mark_completed(run_tag)
            except Exception as e:
                logger.error(f"Phase 2 run {run_tag} FAILED: {e}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))

    ckpt.mark_phase_complete(2)
    logger.info("=== Deep Learning pipeline complete ===")
    logger.info("Phase 1: 8 runs with val metrics → analysis/val_analysis_*.csv")
    logger.info("Phase 2: 2 final runs with test metrics → analysis/final_test_analysis_*.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
    train_dl(args.data_path)
