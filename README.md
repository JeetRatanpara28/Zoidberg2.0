# Zoidberg 2.0 — Pneumonia Detection from Chest X-Ray Images

Epitech MSc IT — Machine Learning Project  
Binary classification: Normal vs Pneumonia

---

## Team

| Person | Role | Models |
|--------|------|--------|
| Jeet (Person A) | Data + Preprocessing + Models | LR, RF, MLP |
| Teammate (Person B) | Splits + Models + Report | SVM, KNN, CNN |

---

## Project Structure

```
zoidberg2/
├── data/
│   ├── train/              ← Dataset 1 (5216 images)
│   ├── val/                ← Dataset 2 (16 images)
│   ├── test/               ← Dataset 3 (624 images)
│   └── preprocessed/       ← Cached preprocessed data
├── notebooks/
│   └── zoidberg2_pneumonia.ipynb
├── models/
│   └── best_pipeline.joblib
├── outputs/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## Setup

### Step 1 — Clone the repo
```bash
git clone <repo-url>
cd zoidberg2
```

### Step 2 — Create virtual environment
```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Open Jupyter
```bash
jupyter notebook
```

---

## For Teammate (Person B) — How to Use Preprocessed Data

Jeet already preprocessed all images and saved them to `data/preprocessed/`.  
You do NOT need to reprocess anything. Just load the cache!

### Step 1 — Open the notebook
Go to `notebooks/zoidberg2_pneumonia.ipynb`

### Step 2 — Run Cell 16 (Imports) first
This loads all libraries and sets dataset paths.

### Step 3 — Add this loading cell and run it

```python
import numpy as np
import joblib

CACHE_DIR = '../data/preprocessed'

# Load preprocessed data
X_train_raw = np.load(f'{CACHE_DIR}/X_train.npy')
y_train     = np.load(f'{CACHE_DIR}/y_train.npy')
X_val_raw   = np.load(f'{CACHE_DIR}/X_val.npy')
y_val       = np.load(f'{CACHE_DIR}/y_val.npy')
X_test_raw  = np.load(f'{CACHE_DIR}/X_test.npy')
y_test      = np.load(f'{CACHE_DIR}/y_test.npy')

# Load fitted scaler and PCA
scaler = joblib.load(f'{CACHE_DIR}/scaler.joblib')
pca    = joblib.load(f'{CACHE_DIR}/pca.joblib')

# Apply transformations
X_train_pca = pca.transform(scaler.transform(X_train_raw))
X_val_pca   = pca.transform(scaler.transform(X_val_raw))
X_test_pca  = pca.transform(scaler.transform(X_test_raw))

print("Data loaded successfully!")
print(f"X_train_pca : {X_train_pca.shape}")
print(f"X_val_pca   : {X_val_pca.shape}")
print(f"X_test_pca  : {X_test_pca.shape}")
```

### Expected output
```
Data loaded successfully!
X_train_pca : (5216, 100)
X_val_pca   : (16, 100)
X_test_pca  : (624, 100)
```

### Step 4 — Also load the splits Jeet prepared

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# Split 1 — Simple Train/Test
X_split1_train, X_split1_test, y_split1_train, y_split1_test = train_test_split(
    X_train_pca, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

# Split 2 — Train/Val/Test (already done by 3 datasets)
X_split2_train = X_train_pca
X_split2_val   = X_val_pca
X_split2_test  = X_test_pca
y_split2_train = y_train
y_split2_val   = y_val
y_split2_test  = y_test

# Split 3 — K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("All splits ready!")
```

### Step 5 — Now train your models (SVM and KNN)

Use `X_split1_train`, `X_split2_train`, `skf` for all 3 splits.  
Use `X_split2_test`, `y_split2_test` for final evaluation.

---

## What Jeet Already Did (Person A)

| Task | Status |
|------|--------|
| Project setup + git | Done |
| EDA — image counts, charts, sample images | Done |
| Preprocessing — grayscale, resize 128x128, normalize, flatten | Done |
| StandardScaler + PCA (100 components, 87.88% variance) | Done |
| Cache saved to data/preprocessed/ | Done |
| Phase 4 — All 3 splits defined | Done |
| Logistic Regression (all 3 splits) | Done |
| Random Forest (all 3 splits) | Done |
| MLP Neural Network — 80% accuracy (all 3 splits) | Done |
| Balanced training with image rotation | Done |
| Best model saved to models/best_pipeline.joblib | Done |

---

## Jeet's Results So Far

| Model | Split | Accuracy | F1 | ROC-AUC |
|-------|-------|----------|-----|---------|
| Logistic Regression | Split 1 | 0.9598 | 0.9728 | 0.9891 |
| Logistic Regression | Split 2 | 0.7484 | 0.8310 | 0.8973 |
| Logistic Regression | Split 3 | 0.9559 | 0.9704 | 0.9871 |
| Random Forest | Split 1 | 0.9416 | 0.9617 | 0.9858 |
| Random Forest | Split 2 | 0.7436 | 0.8287 | 0.9160 |
| Random Forest | Split 3 | 0.9329 | 0.9562 | 0.9833 |
| MLP Neural Network | Split 1 | 0.9617 | 0.9741 | 0.9937 |
| MLP Neural Network | Split 2 | 0.7837 | 0.8512 | 0.9032 |
| MLP Neural Network | Split 3 | 0.9680 | 0.9785 | 0.9928 |

---

## What Teammate Needs to Do (Person B)

| Task | Status |
|------|--------|
| Train SVM — all 3 splits | Todo |
| Train KNN — all 3 splits | Todo |
| GridSearchCV tuning on dataset2 | Todo |
| Final model comparison table | Todo |
| Merge full notebook | Todo |
| Export notebook to HTML | Todo |
| Save best overall model | Todo |
| Write PDF synthesis report | Todo |
| CNN bonus | Optional |
| 3-class prediction bonus | Optional |

---

## Metrics to Compute for Each Model

```python
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score,
                              confusion_matrix, RocCurveDisplay,
                              ConfusionMatrixDisplay)

y_pred = model.predict(X_split2_test)
y_prob = model.predict_proba(X_split2_test)[:,1]

print(f"Accuracy  : {accuracy_score(y_split2_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_split2_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_split2_test, y_pred):.4f}")
print(f"F1        : {f1_score(y_split2_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_split2_test, y_prob):.4f}")
```

---

## How to Use Best Model on New Image

```python
import joblib
import numpy as np
from PIL import Image

# Load saved pipeline
pipeline = joblib.load('../models/best_pipeline.joblib')

# Preprocess new image
img = Image.open('new_xray.jpg').convert('L')
img = img.resize((128, 128))
X_new = np.array(img).flatten() / 255.0
X_new = X_new.reshape(1, -1)

# Predict
result = pipeline.predict(X_new)[0]
prob   = pipeline.predict_proba(X_new)[0]

print("PNEUMONIA" if result == 1 else "NORMAL")
print(f"Confidence: {max(prob):.2%}")
```

---

## Deliverables Checklist

- [ ] zoidberg2_pneumonia.ipynb — complete notebook all cells run
- [ ] zoidberg2_pneumonia.html — exported from notebook
- [ ] summary.pdf — PDF synthesis report
- [ ] models/best_pipeline.joblib — saved best model
- [ ] requirements.txt — all dependencies

---

## Export Notebook to HTML

```bash
jupyter nbconvert --to html notebooks/zoidberg2_pneumonia.ipynb
```