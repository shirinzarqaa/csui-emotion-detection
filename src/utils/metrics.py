import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss

def compute_hierarchical_metrics(y_true, y_pred, id_to_fine, fine_to_basic):
    """
    Menghitung Hierarchical Precision (hP), Hierarchical Recall (hR), dan Hierarchical F1 (hF)
    berdasarkan rumus yang ada di tesis, dimana Set dipanjangkan hingga ke ancestor (label_basic).
    """
    h_precisions = []
    h_recalls = []
    
    for yt, yp in zip(y_true, y_pred):
        # Ancestor extraction for Ground Truth
        true_fine = id_to_fine[yt]
        true_basic = fine_to_basic.get(true_fine, "Unknown")
        true_set = {true_fine, true_basic}
        
        # Ancestor extraction for Prediction
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
    """
    Menjalankan seluruh metrik pelaporan seperti yang tercantum pada BAB II Tesis.
    """
    # 1. Subset Accuracy (Exact Match Ratio)
    # Dalam konteks flat multi-class yang diproksikan ke hierarchical, ini sama dengan Standard Accuracy
    sa = accuracy_score(y_true, y_pred)
    
    # 2. Hamming Loss
    hl = hamming_loss(y_true, y_pred)
    
    # 3. F1-Score (Macro, Micro, Weighted)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    # 4. Hierarchical Metrics
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

def save_manual_analysis(texts, y_true, y_pred, id_to_fine, fine_to_basic, output_path):
    """
    Persiapan untuk *Manual Analysis* oleh peneliti.
    Menyimpan Prediksi versus Aktual beserta hirarkinya (Fine + Basic) dalam sebuah file CSV
    sehingga Anda bisa membaca di mana teks-teks tersebut salah terklasifikasi (error analysis).
    """
    records = []
    for t, yt, yp in zip(texts, y_true, y_pred):
        true_fine = id_to_fine[yt]
        pred_fine = id_to_fine[yp]
        
        status = "Benar Sempurna" if yt == yp else ("Sebagian Benar (Beda Fine, Sama Basic)" if fine_to_basic.get(true_fine) == fine_to_basic.get(pred_fine) else "Salah Total")
        
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
