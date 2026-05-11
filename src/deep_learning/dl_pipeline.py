import argparse
import os
import sys
import gzip
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
from loguru import logger

from src.data_loader import prepare_data
from src.utils.preprocessing import preprocess_for_deep_learning
from src.utils.metrics import compute_all_metrics_binary, save_manual_analysis_binary
from src.deep_learning.models import BiLSTM, TextCNN, FastTextDataset, BertDataset


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


def train_single_run(
    *,
    embedding_type,
    model_type,
    param_value,
    target_level,  # 'basic' or 'fine'
    train_texts, val_texts, test_texts,
    y_train_fine, y_val_fine, y_test_fine,
    y_train_basic, y_val_basic, y_test_basic,
    FINE_TO_ID, ID_TO_FINE,
    BASIC_TO_ID, ID_TO_BASIC,
    FINE_TO_BASIC_TAXONOMY,
    num_epochs=30,
    batch_size=32,
    lr=1e-3,
    patience=3,
    max_len=128,
):
    is_basic = (target_level == 'basic')
    num_classes = len(BASIC_TO_ID) if is_basic else len(FINE_TO_ID)
    id_to_class = ID_TO_BASIC if is_basic else ID_TO_FINE
    class_to_parent = {} if is_basic else FINE_TO_BASIC_TAXONOMY
    y_train = y_train_basic if is_basic else y_train_fine
    y_val = y_val_basic if is_basic else y_val_fine
    y_test = y_test_basic if is_basic else y_test_fine

    run_tag = f"{model_type}_{embedding_type}_{target_level}_{'hid' if model_type == 'bilstm' else 'filt'}{param_value}"
    logger.info(f"Starting run: {run_tag}")

    if embedding_type == 'fasttext':
        word_to_id, emb_matrix, embed_dim = build_fasttext_embedding(train_texts)
        train_ds = FastTextDataset(train_texts, y_train, word_to_id, max_len=max_len)
        val_ds = FastTextDataset(val_texts, y_val, word_to_id, max_len=max_len)
        test_ds = FastTextDataset(test_texts, y_test, word_to_id, max_len=max_len) if len(test_texts) > 0 else None
        bert_mode = False
    elif embedding_type == 'indobert':
        tokenizer, bert_model, embed_dim = build_indobert_embedder(freeze=True)
        train_ds = BertDataset(train_texts, y_train, tokenizer, max_len=max_len)
        val_ds = BertDataset(val_texts, y_val, tokenizer, max_len=max_len)
        test_ds = BertDataset(test_texts, y_test, tokenizer, max_len=max_len) if len(test_texts) > 0 else None
        bert_mode = True
    else:
        raise ValueError(f"Unknown embedding_type: {embedding_type}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size) if test_ds else None

    # Build classifier head
    if model_type == 'bilstm':
        classifier = BiLSTM(embed_dim, hidden_dim=param_value, num_classes=num_classes)
    elif model_type == 'cnn':
        classifier = TextCNN(embed_dim, num_classes=num_classes, num_filters=param_value)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Wrap with embedding if needed
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

    model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
            logger.info(f"  New best val_f1_macro = {best_val_f1:.4f}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"Training complete. Best val_f1_macro = {best_val_f1:.4f}")

    # Final validation evaluation with best model
    _, y_val_true, y_val_pred = evaluate(model, val_loader, criterion, DEVICE, bert_mode=bert_mode)
    final_val_metrics = compute_all_metrics_binary(y_val_true, y_val_pred, id_to_class, class_to_parent)
    for k, v in final_val_metrics.items():
        mlflow.log_metric(f"val_{k}", v)
    if not is_basic:
        y_val_pred_basic = fine_to_basic_predictions(y_val_pred, BASIC_TO_ID, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY)
        final_val_basic = compute_all_metrics_binary(y_val_basic, y_val_pred_basic, ID_TO_BASIC, {})
        for k, v in final_val_basic.items():
            mlflow.log_metric(f"val_basic_{k}", v)

    # Test evaluation
    if test_loader is not None and len(test_texts) > 0:
        _, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, DEVICE, bert_mode=bert_mode)
        test_metrics = compute_all_metrics_binary(y_test_true, y_test_pred, id_to_class, class_to_parent)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        if not is_basic:
            y_test_pred_basic = fine_to_basic_predictions(y_test_pred, BASIC_TO_ID, ID_TO_FINE, FINE_TO_BASIC_TAXONOMY)
            test_basic = compute_all_metrics_binary(y_test_basic, y_test_pred_basic, ID_TO_BASIC, {})
            for k, v in test_basic.items():
                mlflow.log_metric(f"test_basic_{k}", v)

        analysis_dir = "analysis"
        os.makedirs(analysis_dir, exist_ok=True)
        analysis_path = os.path.join(analysis_dir, f"manual_analysis_{run_tag}.csv")
        if is_basic:
            save_manual_analysis_binary(
                test_texts, y_test_true, y_test_pred,
                ID_TO_BASIC, {},
                analysis_path,
                y_true_extra=y_test_fine,
                y_pred_extra=None,
                id_to_extra=ID_TO_FINE,
            )
        else:
            save_manual_analysis_binary(
                test_texts, y_test_true, y_test_pred,
                ID_TO_FINE, FINE_TO_BASIC_TAXONOMY, analysis_path,
            )
        logger.info(f"Manual analysis exported to: {analysis_path}")

        logger.info(
            f"Test results: f1_m={test_metrics['f1_macro']:.4f} | "
            f"sub_acc={test_metrics['subset_accuracy']:.4f} | "
            f"ham_loss={test_metrics['hamming_loss']:.4f}"
        )

        try:
            mlflow.pytorch.log_model(model, "model")
            logger.info("Model artifact logged to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")

    logger.info(f"Run {run_tag} finished.\n")
    return best_val_f1


def train_dl(data_path):
    logger.info("=== Deep Learning Multi-Label Emotion Classification ===")
    logger.info(f"Device: {DEVICE}")

    # Load and preprocess data
    (
        train_df, val_df, test_df,
        y_train_basic, y_val_basic, y_test_basic,
        y_train_fine, y_val_fine, y_test_fine,
        BASIC_TO_ID, ID_TO_BASIC,
        FINE_TO_ID, ID_TO_FINE,
        FINE_TO_BASIC_TAXONOMY,
    ) = prepare_data(data_path)

    logger.info("Preprocessing texts for deep learning ...")
    train_texts = [preprocess_for_deep_learning(t) for t in train_df['text'].values]
    val_texts = [preprocess_for_deep_learning(t) for t in val_df['text'].values]
    test_texts = [preprocess_for_deep_learning(t) for t in test_df['text'].values]
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    y_train_fine = y_train_fine.astype(np.float32)
    y_val_fine = y_val_fine.astype(np.float32)
    y_test_fine = y_test_fine.astype(np.float32)
    y_train_basic = y_train_basic.astype(np.float32)
    y_val_basic = y_val_basic.astype(np.float32)
    y_test_basic = y_test_basic.astype(np.float32)

    mlflow.set_experiment("Deep_Learning_MultiLabel")

    # Best-practice hyperparameters from literature:
    #   BiLSTM: hidden_dim=128 (Graves & Schmidhuber, 2005)
    #   CNN: num_filters=100, filter_sizes=[3,4,5] (Kim, 2014)
    #   Baihaqi et al. (2023) for Indonesian emotion CNN
    embedding_configs = ['fasttext', 'indobert']

    common_kwargs = dict(
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts,
        y_train_fine=y_train_fine,
        y_val_fine=y_val_fine,
        y_test_fine=y_test_fine,
        y_train_basic=y_train_basic,
        y_val_basic=y_val_basic,
        y_test_basic=y_test_basic,
        FINE_TO_ID=FINE_TO_ID,
        ID_TO_FINE=ID_TO_FINE,
        BASIC_TO_ID=BASIC_TO_ID,
        ID_TO_BASIC=ID_TO_BASIC,
        FINE_TO_BASIC_TAXONOMY=FINE_TO_BASIC_TAXONOMY,
    )

    target_levels = ['basic', 'fine']

    for emb_type in embedding_configs:
        for target in target_levels:
            # BiLSTM with fixed hidden_dim=128
            run_tag = f"bilstm_{emb_type}_{target}"
            with mlflow.start_run(run_name=run_tag):
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
                    train_single_run(
                        embedding_type=emb_type,
                        model_type='bilstm',
                        param_value=128,
                        target_level=target,
                        **common_kwargs,
                    )
                except Exception as e:
                    logger.error(f"Run {run_tag} FAILED: {e}")
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))

            # CNN with fixed num_filters=100 (Kim, 2014)
            run_tag = f"cnn_{emb_type}_{target}"
            with mlflow.start_run(run_name=run_tag):
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
                    train_single_run(
                        embedding_type=emb_type,
                        model_type='cnn',
                        param_value=100,
                        target_level=target,
                        **common_kwargs,
                    )
                except Exception as e:
                    logger.error(f"Run {run_tag} FAILED: {e}")
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error", str(e))

    logger.info("=== Deep Learning pipeline complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
    train_dl(args.data_path)