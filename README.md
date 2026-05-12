# Multi-Label Hierarchical Emotion Classification for Indonesian Text

An end-to-end pipeline for hierarchical multi-label emotion classification of Indonesian social media text. Supports three approaches — **Traditional ML**, **Deep Learning**, and **Transformers** — with full experiment tracking via **MLflow**.

---

## System Architecture

```
csui-emotion-detection/
├── data/
│   └── new_all.json          # Primary dataset (32,598 samples)
├── src/
│   ├── data_loader.py        # Multi-label parsing: new_label_basic (7 labels) → new_label_fine_grained (45 labels)
│   ├── utils/
│   │   ├── metrics.py        # Evaluation: f1-macro/micro/weighted, hamming loss, subset accuracy, per-label F1
│   │   └── preprocessing.py  # 3 preprocessing modes: traditional, DL, transformers
│   ├── traditional/
│   │   └── traditional_pipeline.py  # 54 runs: BR/LP × Unigram/Bigram/Trigram × BoW/TF-IDF × LR/NB/SVM
│   ├── deep_learning/
│   │   ├── dl_pipeline.py    # 8 runs: FastText/IndoBERT × BiLSTM/CNN × basic/fine target levels
│   │   └── models.py         # BiLSTM, TextCNN, FastTextDataset, BertDataset
│   └── transformers/
│       └── transformer_pipeline.py  # 40 HPO runs: 5 models × 4 LR × 2 batch sizes
├── analysis/                 # CSV error analysis files after each run
├── run_pipeline.py           # Orchestrator: --run traditional|dl|transformers|all
├── requirements.txt
├── docker-compose.yml        # Parallel Docker: 3 containers + 1 MLflow server
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

Taxonomy built from single-label rows (27,794 of 32,598 data) — each fine-grained label maps to exactly one basic parent.

---

## Data Format

Place `new_all.json` in the `data/` folder (committed to repo, derived from `new_all.xlsx`):

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

Pipeline uses **saka-nlp** (v0.1.9), **emoji** (v2.15+), and **emot** (v3.1) — see [Stack Migration section](#stack-migration-2023--2026) below for rationale.

### URL & Username Patterns in Data

The dataset uses **placeholder tokens** rather than raw URLs/usernames:

| Pattern | Occurrences (out of 32,598) | Example |
|---|---|---|
| `[USERNAME]` | 10,399 rows | `[USERNAME] ini tahun depan wkwk` |
| `[URL]` | 480 rows | `cek [URL] deh` |
| `@username` | 0 rows | — |
| `https?://` | 0 rows | — |
| `#hashtag` | 153 rows | `#IndoPride emang sih` |

Both placeholder format (`[URL]`, `[USERNAME]`) and regex format (`https?://`, `@user`) are handled.

---

## Evaluation Metrics

Each pipeline computes **5 core metrics** + per-label F1:

| Metric | Description |
|---|---|
| **Subset Accuracy (Exact Match Ratio)** | Percentage of samples where all labels match ground truth exactly |
| **Hamming Loss** | Average fraction of incorrectly predicted labels per sample |
| **F1-Macro** | Average F1 across all labels (unweighted) |
| **F1-Micro** | Global F1 (aggregates all labels) |
| **F1-Weighted** | Per-label F1 weighted by label frequency |

Per-label F1 scores are also reported for all 45 fine-grained labels. All metrics are logged to MLflow.

---

## Pipeline: Traditional Machine Learning (54 runs)

### Scenarios

| Scenario | Target | Strategy |
|---|---|---|
| **BR_Basic** | 7 basic labels | Binary Relevance: 7 independent classifiers |
| **LP_Basic** | 7 basic labels | Label Powerset: multi-label combo → single multi-class |
| **BR_Fine** | 45 fine-grained labels | Binary Relevance: 45 independent classifiers |

### Feature Extraction & Classifiers (6 combinations × 3 models = 18 runs per scenario)

| Feature Extraction | Classifier |
|---|---|
| Unigram → Bag of Words | Logistic Regression (max_iter=2000) |
| Unigram → TF-IDF | Multinomial Naive Bayes (alpha=1.0) |
| Bigram → Bag of Words | SVM (kernel=RBF) |
| Bigram → TF-IDF | |
| Trigram → Bag of Words | |
| Trigram → TF-IDF | |

Total: 3 scenarios × 6 features × 3 models = **54 runs**. Models logged to MLflow via `mlflow.sklearn.log_model()`.

---

## Pipeline: Deep Learning (8 runs)

Each model is trained **independently on both target levels** — basic labels (7 classes) and fine-grained labels (45 classes).

### Embedding

Two embedding approaches using best-practice fixed hyperparameters:

| Embedding | Dimension | Source | Rationale |
|---|---|---|---|
| **FastText** (cc.id.300.vec) | 300-dim | Facebook pre-trained Indonesian vectors | Subword info (char n-grams) handles OOV and morphologically rich languages (Conneau et al., 2020). Lightweight, not contextual. |
| **IndoBERT** (indolem/indobert-base-uncased) | 768-dim | IndoLEM pre-trained (Koto et al., 2020) | Contextual embeddings capturing sentence-level semantics. Frozen for feature extraction. |

### Architecture (Fixed Best-Practice Hyperparameters)

| Model | Configuration | Reference |
|---|---|---|
| **Bi-LSTM** | hidden_dim=128, bidirectional, dropout=0.3 | Graves & Schmidhuber (2005) |
| **CNN** | num_filters=100, filter_sizes=[3,4,5], dropout=0.3 | Kim (2014); Baihaqi et al. (2023) for Indonesian |

### Target Levels

| Level | Labels | Output |
|---|---|---|
| **basic** | 7 | Direct 7-label prediction |
| **fine** | 45 | Direct 45-label prediction; basic derived via taxonomy mapping |

Total: 2 embeddings × 2 models × 2 target levels = **8 runs**. Training: BCEWithLogitsLoss, Adam(lr=1e-3), early stopping (patience=3). Models logged to MLflow via `mlflow.pytorch.log_model()`.

---

## Pipeline: Transformers (80 HPO runs)

Each model is fine-tuned **independently on both target levels** — basic labels (7 classes) and fine-grained labels (45 classes).

### HuggingFace Models

| Model | Model ID | Tokenizer |
|---|---|---|
| **IndoBERT (IndoNLU)** | `indobenchmark/indobert-base-p1` | WordPiece |
| **IndoBERT (IndoLEM)** | `indolem/indobert-base-uncased` | WordPiece |
| **IndoBERTweet** | `indolem/indobertweet-base-uncased` | BPE |
| **XLM-R** | `xlm-roberta-base` | SentencePiece BPE |
| **mmBERT** | `jhu-clsp/mmBERT-base` | WordPiece |

### HPO Grid (per target level per model)

- Learning rates: `[2e-5, 3e-5, 4e-5, 5e-5]`
- Batch sizes: `[16, 32]`
- Training: 3 epochs, weight_decay=0.01, `problem_type="multi_label_classification"`

Total: 5 models × 2 target levels × 4 LR × 2 BS = **80 runs**. Checkpoints auto-logged to MLflow via `report_to="mlflow"`.

---

## How to Run

### Docker (Parallel — All Pipelines Simultaneously)

```bash
# Launch MLflow server + 3 pipeline containers in parallel:
docker-compose up --build

# Monitoring:
# - MLflow UI: http://localhost:8002
# - Container logs: docker-compose logs -f
```

This starts **3 independent containers** running in parallel:

| Container | Pipeline | MLflow Experiment | Runs |
|---|---|---|---|
| `pipeline-traditional` | Traditional ML | `Traditional_ML_MultiLabel` | 54 |
| `pipeline-dl` | Deep Learning | `Deep_Learning_MultiLabel` | 8 |
| `pipeline-transformers` | Transformers | `Transformer_MultiLabel` | 80 |

All write to the same MLflow server. Total: **142 concurrent runs**.

### Run a Single Pipeline Only

```bash
docker-compose up traditional       # Traditional ML only
docker-compose up deep-learning     # Deep Learning only
docker-compose up transformers      # Transformers only
```

### Local (without Docker)

```bash
pip install -r requirements.txt
pip install loguru

# All pipelines sequentially:
python run_pipeline.py --data_path ./data/new_all.json --run all

# Individual:
python run_pipeline.py --run traditional
python run_pipeline.py --run dl
python run_pipeline.py --run transformers

# Or directly:
python -m src.traditional.traditional_pipeline --data_path ./data/new_all.json
python -m src.deep_learning.dl_pipeline --data_path ./data/new_all.json
python -m src.transformers.transformer_pipeline --data_path ./data/new_all.json
```

### MLflow Monitoring

```bash
mlflow ui
# Open http://localhost:5000 (local) or http://localhost:8002 (Docker)
```

### Manual Error Analysis

After each run, a CSV is automatically generated in `analysis/` with columns: `Text`, `True_Basic`, `True_Fine`, `Pred_Basic`, `Pred_Fine`, `Status`.

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
```

---

## Stack Migration: 2023 → 2026

The original thesis (Nabila Dita Putri, 2023) used **Sastrawi + manual slang dict + emot**. This pipeline migrates to **saka-nlp + emoji + emot** for the following reasons:

### Summary Table

| Component | Thesis 2023 | Pipeline 2026 | Why Change |
|---|---|---|---|
| **Slang normalization** | Manual JSON dict from `louisowen6/NLP_bahasa_resources` (~1,500 entries, static file) | `saka.normalize()` — 1,018 entries from **Twitter COVID-19 Indonesia corpus** | Lexicon built from real social media data; evolves with Indonesian slang. Manual dict requires self-maintenance and misses newer slang. |
| **Stemming / Morphology** | Sastrawi `StemmerFactory` — suffix-stripping only, returns bare root (`menyebarluaskan` → `sebar`, information lost) | `saka.analyze()` — full morphological analysis: root + prefix + suffix (`menyebarluaskan` → root `sebar luas`, prefixes `['meny']`, suffixes `['kan']`) | Morphological analysis preserves compound-word semantics critical for emotion detection. `sebar luas` ≠ `sebar` in nuance. |
| **Stopword removal** | Sastrawi `StopWordRemoverFactory` — list-based, O(n) lookup per word | `saka.get_stopwords('id')` — **757 words** as `Set`, **O(1) lookup**, based on **Tala Dataset** | On 33K+ texts, O(1) vs O(n) per stopword check is significant. Tala dataset is the community-standard Indonesian stopword list. |
| **Emoji (Unicode) handling** | ❌ **Not handled** — `emot` only handles ASCII emoticons (`:-D`), not Unicode emoji (`🔥😊👍`) | `emoji.demojize(language='id')` — 3,800+ emoji → Indonesian text (`🔥` → `api`, `❤️` → `hati merah`) | The original thesis had **no Unicode emoji handler** — a critical gap for social media text where emoji carry strong emotion signals. |
| **Emoticon handling** | `emot` (same) | `emot` (same) | No change — `emot` still handles ASCII emoticons (`:-)`, `:D`, `:P`). |
| **Tokenization** | `str.split()` — no special handling for punctuation, compound tokens | `saka.tokenize()` — proper Indonesian tokenizer | Better token boundaries = better feature extraction. |
| **Async support** | ❌ Not available | ✅ `async_normalize()`, `async_tokenize()` built-in | Enables non-blocking batch processing on large datasets. |
| **Library maintenance** | Sastrawi: last update **2018** (8 years stagnant). `emot`: stagnant since 2019. | saka-nlp: released **May 2026**. `emoji`: 3.4K+ GitHub stars, actively maintained. | Stale libraries miss new Indonesian slang, morphological patterns, and emoji additions. |

### Performance Impact

| Metric | Thesis 2023 (Sastrawi) | Pipeline 2026 (saka-nlp) | Speedup |
|---|---|---|---|
| Preprocessing time per text | ~181 ms (stemming bottleneck) | ~3.2 ms | **56× faster** |
| Full 32K dataset preprocessing | ~92 min (with LRU cache + `ProcessPoolExecutor`) | ~2 min (single-threaded) | **46× faster** |
| Parallelization needed | Yes (`ProcessPoolExecutor` required) | No (fast enough single-threaded) | Simpler code |

### What Did NOT Change

- **URL & username removal**: Regex-based removal is standard and universal. Only change: added support for `[URL]`/`[USERNAME]` placeholder tokens found in this dataset.
- **Emoticon handling**: Still uses `emot` library — no better alternative exists for ASCII emoticon → text mapping.
- **Hashtag removal**: New addition (not in thesis) — 153 rows in data contain hashtags that are noise for classification.

### References

- **saka-nlp**: [Muhammad-Ikhwan-Fathulloh/Saka-NLP](https://github.com/Muhammad-Ikhwan-Fathulloh/Saka-NLP)
- **emoji**: [carpedm20/emoji](https://github.com/carpedm20/emoji)
- **emot**: [NeelShah18/emot](https://github.com/NeelShah18/emot)
- **Sastrawi** (original): [sastrawi](https://github.com/sastrawi/sastrawi) — last commit 2018

---

## License

MIT — see `LICENSE` file.