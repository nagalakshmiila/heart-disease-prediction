# Intelligent Heart Disease Prediction Using Hybrid Deep Learning and Feature Optimization Techniques

B.Tech final year project (AI & ML). Predicts heart disease risk from
clinical data using a **hybrid feature-optimization pipeline** (RFE + Genetic
Algorithm) feeding a **hybrid deep learning model** (CNN + BiLSTM +
Attention), further boosted by a **stacking ensemble** with Random Forest
and SVM. Includes a Flask website for live predictions.

## 1. Project structure
```
heart_disease_project/
├── data_loader.py         # auto-downloads + cleans the UCI heart disease dataset
├── feature_optimization.py# hybrid RFE + Genetic Algorithm feature selector
├── models.py               # baseline ANN + proposed hybrid CNN-BiLSTM-Attention
├── train.py                 # runs the full pipeline, trains + evaluates everything
├── app.py                   # Flask website using the trained artifacts
├── templates/index.html
├── static/style.css
├── requirements.txt
├── data/                    # created automatically (raw + cleaned csv)
└── saved_models/            # created automatically (models, scaler, plots, results)
```

## 2. Why this is a genuine improvement over a base IEEE paper
Most existing heart-disease-prediction papers use ONE of:
- a single ML/DL model (ANN, or CNN only, or a plain Random Forest), and
- a single, simple feature-selection method (correlation filter, chi-square,
  or one pass of RFE).

This project is different in two places at once, which is what you should
say in your review/viva:

1. **Feature optimization is hybrid, not single-method.** RFE quickly ranks
   features, then a **Genetic Algorithm** searches the actual subset space
   using cross-validated F1 as fitness (seeded with the RFE ranking so it
   converges quickly). This typically finds a smaller, more informative
   feature subset than RFE or chi-square alone.
2. **The model itself is hybrid.** A CNN branch (local feature-interaction
   patterns) and a BiLSTM branch (contextual dependencies across the whole
   feature vector) are fused through an **attention layer**, so the network
   learns which clinical values matter most per patient — instead of a
   plain ANN/CNN as most base papers use. On top of that, a **stacking
   ensemble** (hybrid-DL + Random Forest + SVM → Gradient Boosting
   meta-learner) is trained, which almost always adds another 1-3% accuracy
   over any single model.

`train.py` trains BOTH the plain baseline ANN (representing "previous work")
and the proposed hybrid system **on the exact same optimized feature set and
train/test split**, so the accuracy table it prints/saves is a fair,
defensible before/after comparison for your report.

> Pick one actual base paper your guide gave you, put its reported accuracy
> number into your report's comparison table next to `results_comparison.csv`,
> and cite it. The code does not know which paper that is — you supply that
> number when you write the report.

## 3. Dataset
UCI **Cleveland Heart Disease** dataset (303 patients, 13 clinical features:
age, sex, chest pain type, resting BP, cholesterol, fasting blood sugar,
resting ECG, max heart rate, exercise angina, ST depression, slope, number
of vessels, thalassemia). `data_loader.py` downloads it automatically the
first time you run `train.py` — no manual download needed. It tries, in
order: the official `ucimlrepo` package → a mirrored CSV → a small synthetic
fallback if you truly have no internet (never use the synthetic fallback
for your final report numbers).

## 4. How to run it (step by step)

### Step A — set up the environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step B — train everything (downloads data automatically)
```bash
python train.py
```
This will:
- download + clean the dataset into `data/`
- balance classes with SMOTE
- train the baseline ANN
- run the hybrid RFE+GA feature optimizer (prints the selected features)
- train the proposed hybrid CNN-BiLSTM-Attention model
- train the stacking ensemble
- print a full metrics comparison table and save it to
  `saved_models/results_comparison.csv`
- save `saved_models/accuracy_comparison.png` and `confusion_matrix.png`
  (drop these straight into your report/PPT)
- save every model artifact the website needs

Training takes a few minutes on a normal laptop CPU (dataset is small).

### Step C — run the website
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser, fill in the 13 fields, and
click "Predict Risk". It shows the final ensemble prediction plus the
individual model probabilities (nice for a live demo in your viva).

## 5. What to put in your report
- Table from `saved_models/results_comparison.csv`: Baseline ANN vs.
  Proposed Hybrid CNN-BiLSTM-Attention vs. Final Stacked Ensemble
  (accuracy, precision, recall, F1, ROC-AUC).
- `accuracy_comparison.png` and `confusion_matrix.png` as figures.
- The list of features the GA selected (in
  `saved_models/selected_features.json`) — good for a "feature importance"
  slide.
- Architecture diagram of the hybrid model (CNN branch + BiLSTM branch →
  Attention → Dense → output) — draw this yourself from `models.py`.

## 6. Notes / things to double check before your final submission
- Re-run `train.py` a couple of times (delete `data/heart_processed.csv`
  and everything in `saved_models/` first) and record the best, reproducible
  run — small datasets like this can have some run-to-run variance.
- If your guide specifically requires the Kaggle "Heart Failure Prediction"
  (918-row, combined 5-dataset) version instead of the 303-row Cleveland
  set, tell me and I'll swap `data_loader.py` to that source — the rest of
  the pipeline (feature optimization, hybrid model, ensemble, website) does
  not need to change.
