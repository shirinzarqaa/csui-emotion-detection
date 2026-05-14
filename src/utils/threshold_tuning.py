import numpy as np
from sklearn.metrics import f1_score
from loguru import logger


def optimize_thresholds(y_true, y_probs, id_to_label, step=0.05):
    n_labels = y_true.shape[1]
    thresholds = {}
    for j in range(n_labels):
        label_name = id_to_label.get(j, f"label_{j}")
        best_f1 = -1.0
        best_t = 0.5
        for t in np.arange(0.1, 0.95, step):
            preds = (y_probs[:, j] >= t).astype(np.int32)
            f1 = f1_score(y_true[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
        thresholds[j] = best_t
    return thresholds


def apply_thresholds(y_probs, thresholds):
    n_labels = y_probs.shape[1]
    y_pred = np.zeros_like(y_probs, dtype=np.int32)
    for j in range(n_labels):
        y_pred[:, j] = (y_probs[:, j] >= thresholds.get(j, 0.5)).astype(np.int32)
    return y_pred


def log_thresholds(thresholds, id_to_label, mlflow_log_fn=None):
    for j, t in thresholds.items():
        label_name = id_to_label.get(j, f"label_{j}")
        if mlflow_log_fn:
            mlflow_log_fn(f"threshold_{label_name}", t)
    logger.info(f"Optimized thresholds: mean={np.mean(list(thresholds.values())):.3f}, "
                f"min={min(thresholds.values()):.2f}, max={max(thresholds.values()):.2f}")
