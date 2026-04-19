# HMTC Fine-Grained Emotion Classification Pipeline 🚀

This repository contains a fine-grained emotion classification pipeline utilizing a **Hierarchical Multi-label Text Classification (HMTC)** architecture, specifically designed for Indonesian Text in the Social Media domain. This pipeline supports systematic experiments computed directly into MLFlow for thesis research reproducibility.

## 🌟 Key Features
1. **Multi-Architecture 10-Model Support**: Includes Traditional ML (LR, NB, SVM), Deep Learning (Bi-LSTM, CNN), and Pretrained Language Models (IndoBERT, XLM-R, mmBERT).
2. **Extensive HMTC Evaluation**: Native implementation of *Hierarchical Precision/Recall/F1*, *Hamming Loss*, and *Exact Match Ratio (Subset Accuracy)*.
3. **Manual Error Analysis**: Automatically exports model predictions vs *Ground Truth* into a CSV format ready for human reading and analysis.
4. **Methodological Ablation Validation**: In line with academic journal best practices; supports ablation customization for *N-grams (Unigram, Bigram, Trigram)* and extraction of *Learning Rate Parameters* behavior.

---

## 🛠 Setup and Installation

You can run this pipeline in two ways: using **Docker** (Recommended for stability and automated MLOps tracking) or doing a **Local Python Virtual Environment** installation.

### a. Running via Docker (Fast & Recommended)
Make sure Docker is running on your local machine or server.
```bash
# Automate the setup of both MLFlow Server and the Pipeline Trainer simultaneously
docker-compose up --build
```
> *(Note: This approach ties up the terminal with chained processes. Access the MLFlow link provided in the terminal output to monitor progress via localhost)*

### b. Running via Local (Native Python)
If you aim for extensive modifications directly in the source code or are already using Visual Studio Code Python IDE:
```bash
# 1. Install required packages and modules
pip install -r requirements.txt loguru

# 2. Optional: Install the CUDA variant of PyTorch for NVIDIA GPU acceleration
# (Adjust the pip install pytorch version from the official pytorch site based on your driver)

# 3. Run the main pipeline script (See Section 3 below)
```

---

## 📂 Standard JSON Data Format
This system centrally ingests input data via `.json` files which MUST be placed in `data/dataset.json`.
The internal data structure (a list of dictionaries) must contain at least the following keys:
* `text`: The raw Indonesian social media text.
* `new_label_basic` or `label_basic`: The parent taxonomy label (Example: "Love", "Sadness").
* `new_label_fine_grained` or `label_finegrained`: The deepest branch label (Example: "Nonsexual desire").
* `splitting`: Static data split mapping (*"train", "val", "test"*).

Your data will be parsed automatically (via `src/data_loader.py`), where the models will digest labels dynamically without bounds!

---

## 🚀 Pipeline Execution Tutorial

The exclusive access point to this repository is the `run_pipeline.py` file. You can execute architectures modularly:

### 1. Run All Models Sequentially
```bash
python run_pipeline.py --data_path ./data/dataset.json --run all
```

### 2. Run a Specific Pipeline in Isolation
You can also execute specific pipelines to separate your primary focus if testing iterations take considerable CPU/GPU time:

**A. Traditional ML Pipeline (TF-IDF & N-gram Ablation Study):**
```bash
python run_pipeline.py --data_path ./data/dataset.json --run traditional
```
*Evaluates LogisticRegression, NaiveBayes (Laplace Scaling), and SVM (RBF).*

**B. Deep Learning Pipeline (PyTorch):**
```bash
python run_pipeline.py --data_path ./data/dataset.json --run dl
```
*Performs explicit matrix ablation on Bi-LSTM and CNN (Sentence-level).*

**C. Transformers Pipeline (Hugging Face HPO Target):**
```bash
python run_pipeline.py --data_path ./data/dataset.json --run transformers
```
*Auto-grid hyperparameter optimization for Learning Rates [2e-5, 3e-5, 4e-5, 5e-5] on generic models like IndoBERT.*

---

## 📊 MLFlow Reporting Integration
When the Python scripts above are running, a new directory titled `mlruns` (or `/mlflow_data` if using Docker) will be created to house the robust parameter configuration execution history, recorded in *real-time* from your experiments.

To view the *macro-F1* comparison for all models in a beautiful and interactive graphical web interface:
```bash
# Run in a secondary terminal:
mlflow ui
```
Open the link [http://localhost:5000](http://localhost:5000) on your web browser.

---

## 📚 Manual Model Audit (Error Analysis)
High-standard academic theses always debate not just pure metric calculations but the reality of the classification output (*Error Analysis*). 
This program automatically integrates this into a CSV output tube:
1. Locate the post-execution directory at `analysis/`.
2. Open a file styled like `manual_analysis_SVM_Trigram.csv`.
3. Use *Microsoft Excel / CSV Reader* to study why certain outputs have accuracy statuses reading *"Salah Total (Total Error)"* vs *"Sebagian Benar (Partially Correct)"*. 

Happy writing and experimenting! May this comprehensive analytical system serve as the solid foundation for completing your Thesis! 🎓
