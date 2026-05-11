# Multi-Label Hierarchical Emotion Classification for Indonesian Text

Pipeline klasifikasi emosi hierarkis (multi-label) untuk teks berbahasa Indonesia, mendukung 3 pendekatan: **Traditional ML**, **Deep Learning**, dan **Transformers**. Semua hasil eksperimen tercatat otomatis ke **MLflow**.

---

## Arsitektur Sistem

```
csui-emotion-detection/
├── data/
│   └── new_all.json          # Dataset utama (32.598 sampel)
├── src/
│   ├── data_loader.py        # Parsing multi-label: new_label_basic (7 label) → new_label_fine_grained (45 label)
│   ├── utils/
│   │   ├── metrics.py        # Metrik evaluasi: f1-macro/micro/weighted, hamming loss, subset accuracy, hierarchical P/R/F
│   │   └── preprocessing.py  # 3 mode preprocessing: traditional, DL, transformers
│   ├── traditional/
│   │   └── traditional_pipeline.py  # 54 kombinasi: BR/LP × Unigram/Bigram/Trigram × BoW/TF-IDF × LR/NB/SVM
│   ├── deep_learning/
│   │   ├── dl_pipeline.py    # 12 run: FastText/IndoBERT × BiLSTM/CNN × ablation study
│   │   └── models.py         # BiLSTM, TextCNN, FastTextDataset, BertDataset
│   └── transformers/
│       └── transformer_pipeline.py  # 40 run HPO: 5 model × 4 LR × 2 batch size
├── analysis/                 # CSV hasil manual analysis setelah setiap run
├── run_pipeline.py           # Orchestrator: --run traditional|dl|transformers|all
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## Label Taxonomy (Hierarchical)

| Basic (7) | Fine-grained (45) |
|---|---|
| **anger** | aggressiveness, annoyance, disapproval, disgust, rage |
| **fear** | anticipation, distrust, fear, fears-confirmed, nervousness, restlessness, submission, worry |
| **joy** | acceptance, admiration, amusement, approval, contentment, gratitude, happy-for, joy, optimism, pride, relief, sincerity |
| **love** | attraction, caring, longing, love, lust, nonsexual desire |
| **sadness** | broken-heart, compassion, embarrassment, feeling moved, grief, hopelessness, pensiveness, pity, remorse, suffering |
| **surprise** | confusion, realization, surprise |
| **no emotion** | no emotion |

Taxonomy ini dibangun dari analisis baris single-label (27.794 dari 32.598 data) — setiap fine-grained label memiliki tepat satu parent basic label.

---

## Data Format

Dataset **NOT included** in this repository (`data/*.json` is `.gitignore`d). Provide your own `data/new_all.json` (derived from `new_all.xlsx`):

```json
[
  {
    "id": 1,
    "text": "bang , ada foto rumah nya ga...",
    "new_label_basic": "Love,Sadness",
    "new_label_fine_grained": "Attraction,Suffering",
    "splitting": "train"
  }
]
```

Pipeline reads `new_label_basic` / `new_label_fine_grained` automatically via `src/data_loader.py`.

---

## Preprocessing (3 Mode)

| Step | Traditional | Deep Learning | Transformers |
|---|---|---|---|
| Remove `[URL]` dan URL | ✅ | ❌ | ✅ |
| Remove `[USERNAME]` dan `@user` | ✅ | ❌ | ✅ |
| Ekstraksi emoji → demojize | ✅ | ✅ | ❌ |
| Normalisasi slang | ✅ | ✅ | ✅ |
| Stemming (Sastrawi + LRU cache) | ✅ | ❌ | ❌ |
| Remove stopwords | ✅ | ❌ | ❌ |

Slang dictionary (125+ mappings) dan Indonesian stopwords (200+ words) di `src/utils/preprocessing.py`.

---

## Evaluasi Metrik

Setiap pipeline menghitung **8 metrik evaluasi**:

| Metrik | Deskripsi |
|---|---|
| **Subset Accuracy (Exact Match Ratio)** | Persentase prediksi yang cocok sempurna dengan ground truth (semua label benar) |
| **Hamming Loss** | Rata-rata label yang salah diprediksi per sampel |
| **F1-Macro** | Rata-rata F1 per label (tanpa bobot) |
| **F1-Micro** | F1 global (aggregat semua label) |
| **F1-Weighted** | F1 per label dibobot berdasarkan frekuensi label |
| **Hierarchical Precision** | Precision dengan ekspansi ke parent basic label |
| **Hierarchical Recall** | Recall dengan ekspansi ke parent basic label |
| **Hierarchical F1** | Harmonic mean dari hP dan hR |

Per-label F1 juga dilaporkan untuk setiap fine-grained label. Semua metrik tercatat ke MLflow.

---

## Pipeline: Traditional Machine Learning (54 run)

### Skenario

| Skenario | Target | Strategi |
|---|---|---|
| **BR_Basic** | 7 basic labels | Binary Relevance: 7 classifier independen |
| **LP_Basic** | 7 basic labels | Label Powerset: kombinasi multi-label → single multi-class |
| **BR_Fine** | 45 fine-grained labels | Binary Relevance: 45 classifier independen |

### Fitur & Classifier (6 kombinasi × 3 model = 18 run per skenario)

| Ekstraksi Fitur | Classifier |
|---|---|
| Unigram → Bag of Words | Logistic Regression (max_iter=2000) |
| Unigram → TF-IDF | Multinomial Naive Bayes (alpha=1.0) |
| Bigram → Bag of Words | SVM (kernel=RBF) |
| Bigram → TF-IDF | |
| Trigram → Bag of Words | |
| Trigram → TF-IDF | |

Total: 3 skenario × 6 fitur × 3 model = **54 run**. Model di-log ke MLflow via `mlflow.sklearn.log_model()`.

---

## Pipeline: Deep Learning (12 run)

### Embedding

| Embedding | Dimensi | Karakteristik |
|---|---|---|
| **FastText** (cc.id.300.vec) | 300-dim | Subword information (character n-gram), handles OOV, cocok untuk morfologi Bahasa Indonesia |
| **IndoBERT** (indolem/indobert-base-uncased) | 768-dim | Pre-trained contextual embedding, froze untuk feature extraction |

### Architecture & Ablation

| Model | Variasi Ablation |
|---|---|
| **Bi-LSTM** | hidden_dims = [64, 128, 256] |
| **CNN** | num_filters = [50, 100, 150] |

Total: 2 embedding × (3 BiLSTM + 3 CNN) = **12 run**. Training loop: BCEWithLogitsLoss, Adam(lr=1e-3), early stopping (patience=3). Model di-log ke MLflow via `mlflow.pytorch.log_model()`.

---

## Pipeline: Transformers (40 run HPO)

### Model dari HuggingFace

| Model | Model ID | Tokenizer |
|---|---|---|
| **IndoBERT (IndoNLU)** | `indobenchmark/indobert-base-p1` | WordPiece |
| **IndoBERT (IndoLEM)** | `indolem/indobert-base-uncased` | WordPiece |
| **IndoBERTweet** | `indolem/indobertweet-base-uncased` | BPE |
| **XLM-R** | `xlm-roberta-base` | SentencePiece BPE |
| **mmBERT** | `jhu-clsp/mmBERT-base` | WordPiece |

### HPO Grid

- Learning rates: `[2e-5, 3e-5, 4e-5, 5e-5]`
- Batch sizes: `[16, 32]`
- Training: 3 epoch, weight_decay=0.01, `problem_type="multi_label_classification"`

Total: 5 model × 4 LR × 2 BS = **40 run**. Checkpoints di-log ke MLflow otomatis via `report_to="mlflow"`.

---

## Cara Menjalankan

```bash
# Install dependensi
pip install -r requirements.txt
pip install loguru

# Semua pipeline (54 + 12 + 40 = 106 total run):
python run_pipeline.py --data_path ./data/new_all.json --run all

# Individual:
python run_pipeline.py --run traditional    # 54 run
python run_pipeline.py --run dl              # 12 run
python run_pipeline.py --run transformers    # 40 run

# Atau langsung per pipeline:
python -m src.traditional.traditional_pipeline --data_path ./data/new_all.json
python -m src.deep_learning.dl_pipeline --data_path ./data/new_all.json
python -m src.transformers.transformer_pipeline --data_path ./data/new_all.json
```

### Monitoring MLflow
```bash
mlflow ui
# Buka http://localhost:5000
```

### Analisis Manual
Setelah setiap run, CSV error analysis otomatis di-generate ke folder `analysis/` dengan kolom: `Text`, `True_Basic`, `True_Fine`, `Pred_Basic`, `Pred_Fine`, `Status`.

---

## Dependensi

```
torch, torchvision, torchaudio
transformers>=4.0
datasets
scikit-learn
pandas, numpy
mlflow
Sastrawi     # Indonesian stemmer
emoji        # Emoji handling
nltk         # (listed, not actively used in core pipeline)
tqdm
accelerate
loguru
```

---

## Lisensi
MIT — lihat `LICENSE` file.