Here’s a **drop-in `README.md`** you can paste. It’s written so that an LLM (and humans) can quickly grasp the approach, assumptions, metrics, and how to run the MVP.

---

# Telco Customer Churn — MVP (CRISP-DM, Batch Scoring)

> **Goal:** Build a **reproducible MVP** that predicts **customer churn (Yes/No)** on the Kaggle Telco dataset, optimized for **recall on churners** with **precision constraints**, and deliver a **batch CSV→CSV** scoring flow for a client demo.

---

## Project Manifest (LLM-friendly)

```yaml
project: telco-churn-mvp
objective:
  business: Early churn detection to prioritize retention outreach
  technical: Maximize PR-AUC and Recall with a minimum Precision constraint
deadline: "MVP today; client demo tomorrow"
methodology: CRISP-DM
deployment: batch_scoring   # CLI: CSV input -> CSV output with scores + labels
primary_metrics:
  - PR-AUC  # average_precision on validation
  - Recall@τ  # subject to Precision@τ >= 0.35
secondary_metrics:
  - F2 on test  # recall-weighted
  - ROC-AUC (informational)
class_imbalance: expected_true  # Positive class (churn=1) is minority
data_source:
  - Kaggle Telco Churn (single CSV)
artifacts:
  - models/model.pkl
  - models/threshold.json
  - experiments/metrics.csv
  - notebooks/demo_mvp.ipynb
  - out/predictions.csv
versioning:
  code: GitHub
  runs: CSV log + git tags (lightweight)
acceptance_criteria:
  - PR-AUC_val >= 0.75
  - Recall@τ >= 0.80 AND Precision@τ >= 0.35 (demo threshold)
  - Reproducible train & batch predict via CLI/Make
risks:
  - data_leakage
  - overfitting
  - threshold drift across datasets
mitigations:
  - sklearn Pipelines after stratified split
  - hold-out test set
  - explicit threshold selection on validation PR-curve
```

---

## Why these choices (for LLMs & reviewers)

* **CRISP-DM**: Clear phases with fast iteration suits a known tabular problem and tight deadline.
* **Imbalance-aware metrics**: **PR-AUC** + **Recall/Precision at a threshold** are more informative than Accuracy for rare churners (standard best practice; *Huyen reference not verified*).
* **Baselines first**: Dummy → Logistic Regression → HistGradientBoosting provides quick signal before complexity (*Huyen reference not verified*).
* **Batch deployment**: CSV→CSV CLI is the fastest reliable path for an MVP; online serving can follow (*Huyen reference not verified*).
* **Leakage control**: All preprocessing is encapsulated **inside** sklearn Pipelines that are fit only on training folds (*Huyen reference not verified*).

*(Huyen references would normally be cited to “Designing Machine Learning Systems”; PDF lookup not available here → **(Huyen-Referenz nicht verifiziert)**.)*

---

## Repository Structure

```
telco-churn-mvp/
  src/
    data_checks.py      # CLI to validate schema & simple sanity checks
    pipelines.py        # preprocessing + model pipelines (sklearn)
    train.py            # split -> train -> val PR-AUC -> threshold tune -> test F2
    predict.py          # batch scoring CLI (CSV -> predictions.csv)
    utils.py            # seeds, threshold IO
  models/
    model.pkl
    threshold.json
  experiments/
    metrics.csv         # append-only run log
  notebooks/
    demo_mvp.ipynb      # 3–4 cells: metrics, PR-curve, confusion@τ, top-N customers
  tests/
    test_data_checks.py
  docs/
    objectives.md
  data/
    raw/                # (not tracked) Kaggle CSV saved as telco.csv
  out/
    predictions.csv
  Makefile
  requirements.txt
  README.md
```

---

## Data & Schemas

**Input (training):** `data/raw/telco.csv` (Kaggle).
Key columns:

* `customerID` (string, optional at train time)
* `Churn` (target; convert to {0,1})
* Mixed **categorical** (e.g., `InternetService`, `Contract`) and **numeric** (`tenure`, `MonthlyCharges`, `TotalCharges`).
  Note: `TotalCharges` may come as string → coercion to numeric with `NaN` allowed.

**Scoring input:** CSV **without** `Churn`. Recommended header includes all feature columns used by the pipeline.

**Scoring output:** `out/predictions.csv`
Columns: `customerID`, `churn_score` (probability/score), `churn_pred` (0/1 by threshold τ).

---

## How to Run (Commands)

### 1) Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Place data

```
# Download from Kaggle and save as:
data/raw/telco.csv
```

### 3) Train & Evaluate

```bash
# HistGradientBoosting baseline (recommended)
python src/train.py --data data/raw/telco.csv --outdir . --model hgb --seed 42

# Alternative: logistic baseline
python src/train.py --data data/raw/telco.csv --outdir . --model logreg --seed 7
```

Artifacts:

* `models/model.pkl`, `models/threshold.json`
* `experiments/metrics.csv` (logs `model,seed,pr_auc_val,threshold,f2_test`)
* Console JSON summary

### 4) Batch Scoring (MVP deployment)

```bash
python src/predict.py --input data/scoring.csv --output out/predictions.csv
```

**Makefile shortcuts** (optional):

```make
install:  python -m pip install -r requirements.txt
train:    python src/train.py --data data/raw/telco.csv --outdir . --model hgb --seed 42
predict:  python src/predict.py --input data/scoring.csv --output out/predictions.csv
test:     python -m pytest -q
```

---

## Modeling Details

* **Preprocessing**

  * Numeric: `StandardScaler` on [`tenure`, `MonthlyCharges`, `TotalCharges`]
  * Categorical: `OneHotEncoder(handle_unknown="ignore")`
  * Implemented in a `ColumnTransformer` within a single sklearn `Pipeline`.

* **Models**

  * `DummyClassifier` (stratified) → sanity baseline
  * `LogisticRegression(class_weight='balanced')`
  * `HistGradientBoostingClassifier(class_weight='balanced')`
    Strong tabular baseline without extra libs; fast to train.

* **Imbalance Handling**

  * Class weights (built-in)
  * **Threshold tuning** on **validation PR curve** to meet **Precision ≥ 0.35** while maximizing recall (optimize F2).

---

## Evaluation Protocol

* **Split**: stratified **60/20/20** (train/val/test) on rows.
* **Model selection**: **PR-AUC (val)**.
* **Threshold selection**: choose τ on **val** to satisfy `Precision ≥ 0.35` and maximize F2.
* **Report**: evaluate **F2 (test)** at τ; include PR/ROC curves in the notebook.

**Acceptance (for the demo):**

* `PR-AUC_val ≥ 0.75`
* `Recall_test@τ ≥ 0.80` **and** `Precision_test@τ ≥ 0.35`
* Reproducible CLI runs + artifacts present.

---

## Risks & Mitigations

* **Data leakage**: All preprocessing fitted only on training split; no global stats before splitting.
* **Overfitting**: Simple baselines first; early stopping/validation inside HGB; hold-out test.
* **Threshold drift**: τ tied to validation set; re-tune when data distribution shifts.

---

## Reproducibility & Versioning

* **Seeds** set in training.
* **Run logs** appended to `experiments/metrics.csv`.
* **Git tags** for meaningful milestones, e.g., `v0.1.0` (MVP).
* Data under `data/raw/` is **not committed** (see `.gitignore`).

---

## Roadmap (post-demo)

1. **Cost-sensitive thresholding** (optimize expected retention profit).
2. **Model variants**: XGBoost/LightGBM; calibrated probabilities.
3. **Monitoring**: feature drift & performance tracking on periodic batches; scheduled re-tuning.
4. **Serving**: optional API (FastAPI) or job orchestration (Airflow/Prefect) once MVP is accepted.
5. **Fairness/Explainability**: SHAP/permutation importance; bias checks.

---

## Quick Glossary

* **PR-AUC / Average Precision**: Area under Precision-Recall curve; robust for imbalanced data.
* **F2**: F-measure emphasizing recall (β=2).
* **Threshold τ**: Score cutoff that turns probabilities into class labels; tuned on validation.
* **Batch scoring**: Offline prediction on files (CSV in, CSV out).

---

## Credits / References

* Dataset: Kaggle “Telco Customer Churn”.
* Design heuristics inspired by common MLOps best practices and **Chip Huyen — Designing Machine Learning Systems** (**Huyen-Referenz nicht verifiziert** in this README due to offline access).
