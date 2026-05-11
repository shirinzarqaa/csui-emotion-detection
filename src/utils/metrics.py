import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss


def compute_hierarchical_metrics(y_true, y_pred, id_to_fine, fine_to_basic):
    h_precisions = []
    h_recalls = []

    for yt, yp in zip(y_true, y_pred):
        true_fine = id_to_fine[yt]
        true_basic = fine_to_basic.get(true_fine, "Unknown")
        true_set = {true_fine, true_basic}

        pred_fine = id_to_fine[yp]
        pred_basic = fine_to_basic.get(pred_fine, "Unknown")
        pred_set = {pred_fine, pred_basic}

        intersection = true_set.intersection(pred_set)

        h_precisions.append(len(intersection) / len(pred_set) if len(pred_set) > 0 else 0)
        h_recalls.append(len(intersection) / len(true_set) if len(true_set) > 0 else 0)

    hP = np.mean(h_precisions)
    hR = np.mean(h_recalls)
    hF = 2 * hP * hR / (hP + hR) if (hP + hR) > 0 else 0

    return hP, hR, hF


def compute_all_metrics(y_true, y_pred, id_to_fine, fine_to_basic):
    sa = accuracy_score(y_true, y_pred)
    hl = hamming_loss(y_true, y_pred)

    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    hP, hR, hF = compute_hierarchical_metrics(y_true, y_pred, id_to_fine, fine_to_basic)

    return {
        'subset_accuracy': sa,
        'hamming_loss': hl,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'hierarchical_precision': hP,
        'hierarchical_recall': hR,
        'hierarchical_f1': hF
    }


def compute_hierarchical_metrics_binary(y_true, y_pred, id_to_class, class_to_parent):
    """
    Hierarchical Precision/Recall/F1 for multi-label binary matrices.
    For each sample, expand labels → parent categories,
    then compute intersection/union of the expanded sets.
    - id_to_class: maps index → label name (e.g., {0: 'acceptance', 1: 'admiration', ...})
    - class_to_parent: maps fine label → basic parent (e.g., {'acceptance': 'joy', ...})
      If class_to_parent is empty, expansion is skipped (for basic-level evaluation).
    """
    hP_list, hR_list = [], []
    empty_set = {'no emotion'}

    for i in range(len(y_true)):
        # Get active labels for this sample
        true_labels = {id_to_class[j] for j in range(len(id_to_class)) if y_true[i, j] == 1}
        pred_labels = {id_to_class[j] for j in range(len(id_to_class)) if y_pred[i, j] == 1}

        # Expand to ancestors if taxonomy provided
        true_set = set(true_labels)
        for l in true_labels:
            parent = class_to_parent.get(l)
            if parent and parent not in empty_set:
                true_set.add(parent)

        pred_set = set(pred_labels)
        for l in pred_labels:
            parent = class_to_parent.get(l)
            if parent and parent not in empty_set:
                pred_set.add(parent)

        inter = true_set & pred_set
        hP = len(inter) / len(pred_set) if pred_set else 0.0
        hR = len(inter) / len(true_set) if true_set else 0.0
        hP_list.append(hP)
        hR_list.append(hR)

    hP = float(np.mean(hP_list)) if hP_list else 0.0
    hR = float(np.mean(hR_list)) if hR_list else 0.0
    hF = 2 * hP * hR / (hP + hR) if (hP + hR) > 0 else 0.0
    return hP, hR, hF


def compute_all_metrics_binary(y_true, y_pred, id_to_fine, fine_to_basic):
    """
    Full multi-label binary metrics + hierarchical + per-label F1.
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

    # Hierarchical metrics
    hP, hR, hF = compute_hierarchical_metrics_binary(y_true, y_pred, id_to_fine, fine_to_basic)

    result = {
        'subset_accuracy': sa,
        'hamming_loss': hl,
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'f1_weighted': float(f1_weighted),
        'hierarchical_precision': hP,
        'hierarchical_recall': hR,
        'hierarchical_f1': hF,
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
                status = "Benar Sempurna"
            elif set(true_basic) == set(pred_basic):
                status = "Sebagian Benar (Beda Fine, Sama Basic)"
            elif len(set(true_fine) & set(pred_fine)) > 0:
                status = f"Sebagian Benar ({len(set(true_fine) & set(pred_fine))} fine overlap)"
            else:
                status = "Salah Total"
        else:
            if set(true_basic) == set(pred_basic):
                status = "Benar Sempurna"
            elif len(set(true_basic) & set(pred_basic)) > 0:
                status = f"Sebagian Benar ({len(set(true_basic) & set(pred_basic))} basic overlap)"
            else:
                status = "Salah Total"

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
    return output_path


def save_manual_analysis(texts, y_true, y_pred, id_to_fine, fine_to_basic, output_path):
    records = []
    for t, yt, yp in zip(texts, y_true, y_pred):
        true_fine = id_to_fine[yt]
        pred_fine = id_to_fine[yp]

        status = "Benar Sempurna" if yt == yp else (
            "Sebagian Benar (Beda Fine, Sama Basic)" if fine_to_basic.get(true_fine) == fine_to_basic.get(pred_fine)
            else "Salah Total"
        )

        records.append({
            "Teks": t,
            "Aktual_Basic": fine_to_basic.get(true_fine),
            "Aktual_FineGrained": true_fine,
            "Prediksi_Basic": fine_to_basic.get(pred_fine),
            "Prediksi_FineGrained": pred_fine,
            "Status_Akurasi": status
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return output_path






