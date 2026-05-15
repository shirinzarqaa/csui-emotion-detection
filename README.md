# Multi-Label Hierarchical Emotion Classification for Indonesian Text

An end-to-end pipeline for hierarchical multi-label emotion classification of Indonesian social media text. Supports three approaches — **Traditional ML**, **Deep Learning**, and **Transformers** — across **two experiments** (baseline vs. optimized), with full experiment tracking via **MLflow** and comprehensive comparison analysis.

---

## System Architecture

```
csui-emotion-detection/
├── data/
│   └── new_all.json                    # Primary dataset (32,598 samples)
├── src/
│   ├── data_loader.py                  # Multi-label parsing: basic (6) → fine-grained (44)
│   ├── utils/
│   │   ├── metrics.py                  # 5 core metrics + per-label precision/recall/F1/support + per-sample hamming/exact-match
│   │   ├── preprocessing.py            # 3 preprocessing modes: traditional, DL, transformers
│   │   ├── inference_analysis.py       # Deep per-sample inference analysis (Phase 2 only)
│   │   ├── experiment_config.py        # Central config: exp1 (baseline) vs exp2 (optimized)
│   │   ├── mlflow_utils.py             # Safe MLflow wrapper: 5 retries, 10s incremental delay, fallback
│   │   ├── threshold_tuning.py         # Per-label threshold optimization (0.1–0.9, step 0.05)
│   │   ├── focal_loss.py               # FocalLoss(nn.Module): alpha (pos_weight) + gamma=2.0
│   │   ├── generate_comparison.py      # Master comparison spreadsheet (9+ sheets: exp1 vs exp2)
│   │   └── checkpoint.py               # CheckpointManager: resume/skip completed runs
│   ├── traditional/
│   │   └── traditional_pipeline.py     # 57 runs: BR/LP × features × classifiers + threshold tuning (exp2)
│   ├── deep_learning/
│   │   ├── dl_pipeline.py              # 10 runs: FastText/IndoBERT × BiLSTM/CNN × basic/fine + FocalLoss (exp2)
│   │   └── models.py                   # BiLSTM, TextCNN, FastTextDataset, BertDataset
│   └── transformers/
│       └── transformer_pipeline.py     # 40 HPO runs: 5 models × 2 targets × 3 LR × 1 BS + WeightedTrainer (exp2)
├── analysis/                           # Exp1: CSV/XLSX error analysis (val + test)
├── analysis_exp2/                      # Exp2: CSV/XLSX error analysis (val + test)
├── saved_models/                        # Exp1: saved model artifacts per run
├── saved_models_exp2/                   # Exp2: saved model artifacts per run
├── checkpoints/                         # *_checkpoint.json (exp1) + *_checkpoint_exp2.json (exp2)
├── run_pipeline.py                      # Orchestrator: --run traditional|dl|transformers|all --experiment exp1|exp2
├── requirements.txt
├── docker-compose.yml                   # Parallel Docker: 3 containers + 1 MLflow server
└── README.md
```

---

## Label Taxonomy (Hierarchical)

| Basic (6) | Fine-grained (44) |
|---|---|
| **anger** | aggressiveness, annoyance, disapproval, disgust, rage |
| **fear** | anticipation, distrust, fear, fears-confirmed, nervousness, restlessness, submission, worry |
| **joy** | acceptance, admiration, amusement, approval, contentment, gratitude, happy-for, joy, optimism, pride, relief, sincerity |
| **love** | attraction, caring, longing, love, lust, nonsexual desire |
| **sadness** | broken-heart, compassion, embarrassment, feeling moved, grief, hopelessness, pensiveness, pity, remorse, suffering |
| **surprise** | confusion, realization, surprise |

> **Note — "no emotion" removed**: The original dataset includes a "no emotion" label, but it has **0 training examples** (only 4 in val, 6 in test). Since no model can learn a class with zero training examples, it has been removed from the label set.

---

## Two Experiments: Baseline vs. Optimized

| | Exp1 (Baseline) | Exp2 (Optimized) |
|---|---|---|
| **Purpose** | Standard training, default settings | Address class imbalance + improve fine-grained F1 |
| **Epochs** | 3 (transformers/DL) | 10 (DL), 5 (transformers) with early stopping |
| **Max length** | 128 | 256 |
| **Loss function** | BCEWithLogitsLoss | FocalLoss (gamma=2.0) + pos_weight (clamped [1.0, 10.0]) |
| **Threshold** | 0.5 (default) | Per-label optimized (0.1–0.9, step 0.05) |
| **BiLSTM capacity** | hidden=128, layers=1 | hidden=256, layers=2 |
| **MLflow experiment** | `Traditional_ML_MultiLabel`, etc. | `Traditional_ML_MultiLabel_exp2`, etc. |
| **Checkpoints** | `checkpoints/*_checkpoint.json` | `checkpoints/*_checkpoint_exp2.json` |
| **Analysis output** | `analysis/` | `analysis_exp2/` |
| **Saved models** | `saved_models/` | `saved_models_exp2/` |

### Exp2 Key Optimizations

1. **Focal Loss** (`src/utils/focal_loss.py`): Down-weights easy examples, focuses training on hard/rare labels. Alpha from pos_weight + gamma=2.0.
2. **Pos Weight** (`compute_pos_weights()`): Inversely proportional to class frequency, clamped to [1.0, 10.0] to prevent explosion. Applied in loss function for all three pipelines.
3. **Per-Label Threshold Tuning** (`src/utils/threshold_tuning.py`): Sweeps 0.1–0.9 (step 0.05) per label, picks threshold maximizing F1 for that label. Replaces the default 0.5 sigmoid cutoff.
4. **WeightedTrainer** (transformers): Custom HuggingFace Trainer subclass overriding `compute_loss()` to inject pos_weight + FocalLoss.
5. **Larger BiLSTM**: hidden_dim=256, num_layers=2 for fine-grained (44 labels) capacity.

---

## Data Format

Place `new_all.json` in the `data/` folder:

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

## Preprocessing (3 Modes)

| Step | Traditional | Deep Learning | Transformers |
|---|---|---|---|
| Case folding (lowercase) | ✓ | ✓ | ✓ |
| Remove `[URL]` / `https?://` | ✓ | ✗ | ✓ |
| Remove `[USERNAME]` / `@user` | ✓ | ✗ | ✓ |
| Remove `#hashtag` | ✓ | ✗ | ✓ |
| Emoji → Indonesian text (`emoji.demojize(language='id')`) | ✓ | ✓ | ✗ |
| Emoticon → text (`emot`) | ✓ | ✓ | ✗ |
| Slang normalization (`saka.normalize()`) | ✓ | ✓ | ✓ |
| Tokenization (`saka.tokenize()`) | ✓ | ✓ | ✓ |
| Morphological analysis (`saka.analyze()`) | ✓ | ✗ | ✗ |
| Remove stopwords (`saka.get_stopwords('id')`, 757 words) | ✓ | ✗ | ✗ |
| Remove non-alphabetic tokens | ✓ | ✓ | ✓ |
| Clean extra spaces | ✓ | ✓ | ✓ |

Pipeline uses **saka-nlp** (v0.1.9), **emoji** (v2.15+), and **emot** (v3.1).

### URL & Username Patterns in Data

| Pattern | Occurrences (out of 32,598) | Example |
|---|---|---|
| `[USERNAME]` | 10,399 rows | `[USERNAME] ini tahun depan wkwk` |
| `[URL]` | 480 rows | `cek [URL] deh` |
| `@username` | 0 rows | — |
| `https?://` | 0 rows | — |
| `#hashtag` | 153 rows | `#IndoPride emang sih` |

---

## Evaluation Metrics

Each pipeline computes **5 core metrics** + per-label metrics:

| Metric | Description |
|---|---|
| **Subset Accuracy (Exact Match Ratio)** | Percentage of samples where all labels match ground truth exactly |
| **Hamming Loss** | Average fraction of incorrectly predicted labels per sample |
| **F1-Macro** | Average F1 across all labels (unweighted) |
| **F1-Micro** | Global F1 (aggregates all labels) |
| **F1-Weighted** | Per-label F1 weighted by label frequency |

Additionally: per-label **precision**, **recall**, **F1**, **support** for all labels, plus per-sample **hamming loss** and **exact match** flag. All metrics are logged to MLflow.

---

## Experiment Methodology: Phase 1 + Phase 2

To prevent **test set leakage**, all pipelines follow a strict two-phase protocol:

> **Rule: Test set is touched ONCE, at the very end. Val set is used during development.**

### Phase 1: Experimentation (many runs)

| Step | Dataset | Purpose |
|---|---|---|
| Fit model | Train | Learn parameters |
| Predict + metrics | **Val** | Compare models, select best config |
| Log to MLflow | Val metrics | Track experiments |
| Error analysis CSV+XLSX | Val | `analysis/val_analysis_*.csv` + `.xlsx` |
| Test? | **SKIP** | Never during development |

### Phase 2: Final Report (once, at the end)

| Step | Dataset | Purpose |
|---|---|---|
| Retrain best model | **Train + Val** | Maximize training data |
| Predict + metrics | **Test** | Unbiased final score (once) |
| Error analysis CSV+XLSX | Test | `analysis/final_test_analysis_*.csv` + `.xlsx` |
| Log to MLflow | Test metrics | Archive final result |

### MLflow Organization

| Experiment | Phase | Exp1 Name | Exp2 Name | Runs |
|---|---|---|---|---|
| Traditional ML | Phase 1 | `Traditional_ML_MultiLabel` | `Traditional_ML_MultiLabel_exp2` | 54 |
| Traditional ML | Phase 2 | `Traditional_ML_Final_Test` | `Traditional_ML_Final_Test_exp2` | 3 |
| Deep Learning | Phase 1 | `Deep_Learning_MultiLabel` | `Deep_Learning_MultiLabel_exp2` | 8 |
| Deep Learning | Phase 2 | `Deep_Learning_Final_Test` | `Deep_Learning_Final_Test_exp2` | 2 |
| Transformers | Phase 1 | `Transformer_MultiLabel` | `Transformer_MultiLabel_exp2` | 30 |
| Transformers | Phase 2 | `Transformer_Final_Test` | `Transformer_Final_Test_exp2` | 10 |

---

## Pipeline: Traditional Machine Learning (57 runs total)

### Scenarios

| Scenario | Target | Strategy |
|---|---|---|
| **BR_Basic** | 6 basic labels | Binary Relevance: 6 independent classifiers |
| **LP_Basic** | 6 basic labels | Label Powerset: multi-label combo → single multi-class |
| **BR_Fine** | 44 fine-grained labels | Binary Relevance: 44 independent classifiers |

### Feature Extraction & Classifiers (6 combinations × 3 models = 18 runs per scenario)

| Feature Extraction | Classifier |
|---|---|
| Unigram → Bag of Words | Logistic Regression (max_iter=2000) |
| Unigram → TF-IDF | Multinomial Naive Bayes (alpha=1.0) |
| Bigram → Bag of Words | SVM (kernel=RBF) |
| Bigram → TF-IDF | |
| Trigram → Bag of Words | |
| Trigram → TF-IDF | |

Total: 3 scenarios × 6 features × 3 models = **54 Phase 1 runs** + **3 Phase 2 runs** = **57 total**.

### Exp2 additions
- **Threshold tuning** for BR models (LR/SVM via `predict_proba`): sweeps 0.1–0.9 per label, replaces default 0.5.

---

## Pipeline: Deep Learning (10 runs total)

Each model is trained **independently on both target levels** — basic labels (6 classes) and fine-grained labels (44 classes).

### Embedding

| Embedding | Dimension | Source | Rationale |
|---|---|---|---|
| **FastText** (cc.id.300.vec) | 300-dim | Facebook pre-trained Indonesian vectors | Subword info handles OOV and morphologically rich languages. Lightweight, not contextual. |
| **IndoBERT** (indolem/indobert-base-uncased) | 768-dim | IndoLEM pre-trained | Contextual embeddings. Frozen for feature extraction. |

### Architecture

| Model | Exp1 Config | Exp2 Config |
|---|---|---|
| **Bi-LSTM** | hidden_dim=128, 1 layer, dropout=0.3 | hidden_dim=256, 2 layers, dropout=0.3 |
| **CNN** | num_filters=100, filter_sizes=[3,4,5], dropout=0.3 | (same) |

### Target Levels

| Level | Labels | Output |
|---|---|---|
| **basic** | 6 | Direct 6-label prediction |
| **fine** | 44 | Direct 44-label prediction; basic derived via taxonomy mapping |

Total: 2 embeddings × 2 models × 2 target levels = **8 Phase 1 runs** + **2 Phase 2 runs** = **10 total**.

### Exp2 additions
- **FocalLoss** (gamma=2.0) replaces BCEWithLogitsLoss
- **pos_weight** computed from training data, clamped [1.0, 10.0]
- **Threshold tuning** per label after prediction
- **Epochs**: 10 (up from 3) with early stopping (patience=3)

---

## Pipeline: Transformers (40 HPO runs total)

Each model is fine-tuned **independently on both target levels** — basic labels (6 classes) and fine-grained labels (44 classes).

### HuggingFace Models

| Model | Model ID | Tokenizer |
|---|---|---|
| **IndoBERT (IndoNLU)** | `indobenchmark/indobert-base-p1` | WordPiece |
| **IndoBERT (IndoLEM)** | `indolem/indobert-base-uncased` | WordPiece |
| **IndoBERTweet** | `indolem/indobertweet-base-uncased` | BPE |
| **XLM-R** | `xlm-roberta-base` | SentencePiece BPE |
| **mmBERT** | `jhu-clsp/mmBERT-base` | WordPiece |

### HPO Grid (per target level per model)

- Learning rates: `[2e-5, 3e-5, 5e-5]`
- Batch sizes: `[32]`
- Training: 5 epochs (exp2) / 3 epochs (exp1), weight_decay=0.01, early stopping (patience=3)

Total: 5 models × 2 target levels × 3 LR × 1 BS = **30 Phase 1 runs** + **10 Phase 2 runs** = **40 total**.

### Exp2 additions
- **WeightedTrainer**: Custom Trainer subclass with FocalLoss + pos_weight in `compute_loss()`
- **pos_weight** computed from training data, clamped [1.0, 10.0]
- **Threshold tuning** per label after prediction
- **save_total_limit=1**: prevents disk overflow from accumulating checkpoints

---

## Experiment Comparison & Thesis Results

### Thesis Results (Val + Test per run)

Evaluate **all Phase 1 models on test set** (models trained on train only, no retrain on train+val):

```bash
# Evaluate all models on test set
python -m src.utils.evaluate_test --experiment exp1 --output thesis_results
python -m src.utils.evaluate_test --experiment exp2 --output thesis_results

# Generate combined thesis tables
python -m src.utils.thesis_results --output thesis_results --eval_dir thesis_results
```

**Output: 2 separate spreadsheets (exp1 + exp2), each with all runs and both val & test metrics:**

| Column | Description |
|---|---|
| Percobaan | Run name (e.g. `BR_Basic_Unigram_BoW_LR`, `bilstm_fasttext_basic`, `IndoBERT_basic_lr3e-05_bs32`) |
| Val F1-Micro | F1-micro on validation set |
| Val F1-Macro | F1-macro on validation set |
| Val F1-Weighted | F1-weighted on validation set |
| Val Hamming Loss | Hamming loss on validation set |
| Val EMR | Exact Match Ratio (subset accuracy) on validation set |
| Test F1-Micro | F1-micro on test set |
| Test F1-Macro | F1-macro on test set |
| Test F1-Weighted | F1-weighted on test set |
| Test Hamming Loss | Hamming loss on test set |
| Test EMR | Exact Match Ratio (subset accuracy) on test set |

**Files:**
- `thesis_results/exp1_all_val_test.xlsx` — Experiment 1 (baseline), all runs with val + test
- `thesis_results/exp2_all_val_test.xlsx` — Experiment 2 (optimized), all runs with val + test
- `thesis_results/exp1_all_results.xlsx` + `exp2_all_results.xlsx` — Combined MLflow + evaluate_test results
- `thesis_results/thesis_results.xlsx` — Master file with summary sheet

### Master Comparison Spreadsheet

```bash
python -m src.utils.generate_comparison --output experiment_comparison
```

**Output** (9+ sheets in XLSX + CSVs):

| Sheet | Contents |
|---|---|
| exp1_val_summary | All Phase 1 val metrics for exp1 |
| exp2_val_summary | All Phase 1 val metrics for exp2 |
| exp1_test_summary | All Phase 2 test metrics for exp1 |
| exp2_test_summary | All Phase 2 test metrics for exp2 |
| comparison | Side-by-side val + test F1 delta (exp2 - exp1) |
| thresholds | Per-label optimized thresholds for all exp2 runs |
| per_label_f1 | Per-label F1 for all runs |
| per_label_comparison | Per-label F1 delta (exp2 - exp1) |
| emr_per_run | Exact Match Ratio per run |
| f1_per_basic_group | F1 grouped by basic emotion category |

---

## How to Run

### Docker (Parallel — All Pipelines Simultaneously)

```bash
docker-compose up --build

# Monitoring:
# - MLflow UI: http://localhost:8002
# - Container logs: docker-compose logs -f
```

| Container | Pipeline | Exp1 Phase 1 | Exp2 Phase 1 | Total |
|---|---|---|---|---|
| `pipeline-traditional` | Traditional ML | 54 | 54 | 108 |
| `pipeline-dl` | Deep Learning | 8 | 8 | 16 |
| `pipeline-transformers` | Transformers | 30 | 30 | 60 |

Grand total (Phase 1 + Phase 2, both experiments): **184 + 30 = 214 runs**.

### Run a Single Pipeline Only

```bash
docker-compose up traditional
docker-compose up deep-learning
docker-compose up transformers
```

### Local (without Docker)

```bash
pip install -r requirements.txt

# Exp1 (baseline):
python run_pipeline.py --run traditional --experiment exp1
python run_pipeline.py --run dl --experiment exp1
python run_pipeline.py --run transformers --experiment exp1

# Exp2 (optimized):
python run_pipeline.py --run traditional --experiment exp2
python run_pipeline.py --run dl --experiment exp2
python run_pipeline.py --run transformers --experiment exp2

# All pipelines sequentially (default: exp1):
python run_pipeline.py --run all
```

### MLflow Monitoring

```bash
mlflow ui
# Open http://localhost:5000 (local) or http://localhost:8002 (Docker)
```

### Error Analysis

**Phase 1 (val)**: After each run, CSV+XLSX in `analysis/val_analysis_*` (exp1) or `analysis_exp2/val_analysis_*` (exp2).

**Phase 2 (test)**: After final evaluation, CSV+XLSX in `analysis/final_test_analysis_*` (exp1) or `analysis_exp2/final_test_analysis_*` (exp2).

### Deep Inference Analysis (Phase 2 only)

```python
from src.utils.inference_analysis import run_inference_and_analysis

df = run_inference_and_analysis(
    y_pred=y_pred_binary,
    y_prob=probs_matrix,
    y_test=y_test_binary,
    texts=test_texts_raw,
    id_to_label=ID_TO_BASIC,
    pipeline_name="bilstm_fasttext_basic",
)
```

**Output files:**

| File | Contents |
|---|---|
| `deep_analysis_*.csv` + `.xlsx` | Per-sample: text, true/pred labels, confidence, status, prob per label |
| `hard_samples_*.csv` + `.xlsx` | Subset: complete_mismatch + low confidence + ambiguous labels |
| `deep_analysis_summary_*.txt` | Aggregate: status distribution, avg confidence, per-label avg probability |

---

## Checkpoint & Resume

All pipelines support automatic resume via `CheckpointManager`:

- Completed runs are logged to `checkpoints/*_checkpoint.json` (exp1) or `checkpoints/*_checkpoint_exp2.json` (exp2)
- On restart, completed runs are **skipped automatically**
- Training resumes from the first unfinished run

---

## Dependencies

```
torch, torchvision, torchaudio
transformers>=4.0
datasets
scikit-learn
pandas, numpy
mlflow
saka-nlp          # Indonesian NLP: slang normalization, morphological analysis, stopwords, tokenization
emoji             # Emoji Unicode → text (language='id')
emot              # ASCII emoticon → text
tqdm
accelerate
loguru
openpyxl          # XLSX output for analysis spreadsheets
```

---

## Stack Migration: 2023 → 2026

The original thesis (Nabila Dita Putri, 2023) used **Sastrawi + manual slang dict + emot**. This pipeline migrates to **saka-nlp + emoji + emot**:

### Summary Table

| Component | Thesis 2023 | Pipeline 2026 | Why Change |
|---|---|---|---|
| **Slang normalization** | Manual JSON dict (~1,500 entries, static) | `saka.normalize()` — 1,018 entries from **Twitter COVID-19 Indonesia corpus** | Lexicon built from real social media data; evolves with Indonesian slang. |
| **Stemming / Morphology** | Sastrawi `StemmerFactory` — suffix-stripping only | `saka.analyze()` — full morphological analysis: root + prefix + suffix | Morphological analysis preserves compound-word semantics critical for emotion detection. |
| **Stopword removal** | Sastrawi `StopWordRemoverFactory` — O(n) lookup | `saka.get_stopwords('id')` — **757 words** as `Set`, **O(1) lookup** | On 33K+ texts, O(1) vs O(n) per stopword check is significant. |
| **Emoji (Unicode) handling** | ❌ **Not handled** | `emoji.demojize(language='id')` — 3,800+ emoji → Indonesian text | Critical gap for social media text where emoji carry strong emotion signals. |
| **Emoticon handling** | `emot` (same) | `emot` (same) | No change. |
| **Tokenization** | `str.split()` | `saka.tokenize()` | Better token boundaries = better feature extraction. |

### Performance Impact

| Metric | Thesis 2023 (Sastrawi) | Pipeline 2026 (saka-nlp) | Speedup |
|---|---|---|---|
| Preprocessing time per text | ~181 ms | ~3.2 ms | **56× faster** |
| Full 32K dataset preprocessing | ~92 min | ~2 min | **46× faster** |

### References

- **saka-nlp**: [Muhammad-Ikhwan-Fathulloh/Saka-NLP](https://github.com/Muhammad-Ikhwan-Fathulloh/Saka-NLP)
- **emoji**: [carpedm20/emoji](https://github.com/carpedm20/emoji)
- **emot**: [NeelShah18/emot](https://github.com/NeelShah18/emot)
- **Sastrawi** (original): [sastrawi](https://github.com/sastrawi/sastrawi) — last commit 2018

---

## License

MIT — see `LICENSE` file.
