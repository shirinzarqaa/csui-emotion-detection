import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss


def compute_all_metrics_binary(y_true, y_pred, id_to_fine, fine_to_basic):
    """
    Full multi-label binary metrics: subset accuracy, hamming loss,
    f1-macro/micro/weighted, and per-label F1 scores.
    y_true, y_pred: (N, C) binary numpy arrays.
    """
    from sklearn.metrics import f1_score

    N, C = y_true.shape

    # Subset accuracy (exact match of all labels per sample)
    sa = float(np.mean(np.all(y_true == y_pred, axis=1)))

    # Hamming loss
    hl = float(np.sum(y_true != y_pred) / (N * C))

    # F1 per label
    per_label_f1 = {}
    for j in range(C):
        label_name = id_to_fine.get(j, f"label_{j}")
        f1 = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
        per_label_f1[f"f1_{label_name}"] = float(f1)

    # Macro / Micro / Weighted F1 (scikit-learn multi-label)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    result = {
        'subset_accuracy': sa,
        'hamming_loss': hl,
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
    }
    result.update(per_label_f1)
    return result


def save_manual_analysis_binary(texts, y_true, y_pred, id_to_class, class_to_parent, output_path,
                              y_true_extra=None, y_pred_extra=None, id_to_extra=None):
    """
    Multi-label manual analysis CSV.
    - For fine-grained labels: y_true/y_pred are fine binary matrices, id_to_class=fine map, class_to_parent=fine→basic taxonomy
    - For basic labels: y_true/y_pred are basic binary matrices, id_to_class=basic map, class_to_parent=empty dict
    - If y_true_extra/y_pred_extra/id_to_extra are provided, they reference the other label level for richer status
    """
    is_fine_level = bool(class_to_parent)
    records = []

    for i, text in enumerate(texts):
        true_labels = sorted([id_to_class[j] for j in range(len(id_to_class)) if y_true[i, j] == 1])
        pred_labels = sorted([id_to_class[j] for j in range(len(id_to_class)) if y_pred[i, j] == 1])

        if is_fine_level:
            true_basic = sorted(set(class_to_parent.get(l, '?') for l in true_labels))
            pred_basic = sorted(set(class_to_parent.get(l, '?') for l in pred_labels))
            true_fine = true_labels
            pred_fine = pred_labels
        else:
            true_basic = true_labels
            pred_basic = pred_labels
            true_fine = []
            pred_fine = []
            if y_true_extra is not None and y_pred_extra is not None and id_to_extra:
                true_fine = sorted([id_to_extra[j] for j in range(len(id_to_extra)) if y_true_extra[i, j] == 1])
                pred_fine = sorted([id_to_extra[j] for j in range(len(id_to_extra)) if y_pred_extra[i, j] == 1])

        if is_fine_level:
            if set(true_fine) == set(pred_fine):
                status = "Exact Match"
            elif set(true_basic) == set(pred_basic):
                status = "Partial Match (same basic, different fine)"
            elif len(set(true_fine) & set(pred_fine)) > 0:
                status = f"Partial Match ({len(set(true_fine) & set(pred_fine))} fine overlap)"
            else:
                status = "Complete Mismatch"
        else:
            if set(true_basic) == set(pred_basic):
                status = "Exact Match"
            elif len(set(true_basic) & set(pred_basic)) > 0:
                status = f"Partial Match ({len(set(true_basic) & set(pred_basic))} basic overlap)"
            else:
                status = "Complete Mismatch"

        records.append({
            'Text': text,
            'True_Basic': ', '.join(true_basic) if true_basic else 'none',
            'True_Fine': ', '.join(true_fine) if true_fine else 'none',
            'Pred_Basic': ', '.join(pred_basic) if pred_basic else 'none',
            'Pred_Fine': ', '.join(pred_fine) if pred_fine else 'none',
            'Status': status,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    xlsx_path = output_path.rsplit('.', 1)[0] + '.xlsx'
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
    return output_path


def save_manual_analysis(texts, y_true, y_pred, id_to_fine, fine_to_basic, output_path):
    records = []
    for t, yt, yp in zip(texts, y_true, y_pred):
        true_fine = id_to_fine[yt]
        pred_fine = id_to_fine[yp]

        status = "Exact Match" if yt == yp else (
            "Partial Match (same basic, different fine)" if fine_to_basic.get(true_fine) == fine_to_basic.get(pred_fine)
            else "Complete Mismatch"
        )

        records.append({
            "Text": t,
            "Actual_Basic": fine_to_basic.get(true_fine),
            "Actual_FineGrained": true_fine,
            "Predicted_Basic": fine_to_basic.get(pred_fine),
            "Predicted_FineGrained": pred_fine,
            "Status_Accuracy": status
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    xlsx_path = output_path.rsplit('.', 1)[0] + '.xlsx'
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
    return output_path

