# Zoidberg 2.0 — Pneumonia Detection from Chest X-Ray Images

Epitech MSc IT — Machine Learning Project

---

## Project Structure

```
Zoidberg2.0/
├── data/
│   ├── train/              ← 5,216 images (NORMAL + PNEUMONIA)
│   ├── val/                ← 16 images
│   ├── test/               ← 624 images
│   └── preprocessed/       ← cached .npy files (scaler + PCA)
├── notebooks/
│   └── zoidberg2_pneumonia.ipynb
├── models/
│   ├── best_pipeline.joblib
│   ├── svm_best.joblib
│   ├── knn_best.joblib
│   ├── cnn_best.keras
│   ├── svm3_best.joblib
│   ├── knn3_best.joblib
│   └── cnn3_best.keras
├── outputs/
│   ├── figures/            ← all plots
│   ├── synthesis_report.pdf
│   └── zoidberg2_pneumonia_full.html
├── run_models.py
├── run_3class.py
├── make_report.py
└── requirements.txt
```

---

## Setup

```bash
# Create virtual environment (Python 3.11 required for TensorFlow)
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset

| Split | Folder | Normal | Pneumonia | Total |
|-------|--------|--------|-----------|-------|
| Train | data/train | 1,341 | 3,875 | 5,216 |
| Val   | data/val   | 8     | 8         | 16    |
| Test  | data/test  | 234   | 390       | 624   |

---

## Preprocessing

All images go through:
1. Grayscale conversion — `.convert('L')`
2. Resize to 128×128
3. Flatten to 1D — `.flatten()`
4. Normalize — divide by 255.0
5. StandardScaler (fit on train only)
6. PCA — 100 components, 87.88% variance retained

CNN uses raw 64×64 grayscale pixels (no PCA).

---

## Models & Results

### Binary Classification (Normal vs Pneumonia)

| Model | Split | Accuracy | F1 | ROC-AUC |
|-------|-------|----------|----|---------|
| Logistic Regression | Split 1 | 0.9598 | 0.9728 | 0.9891 |
| Logistic Regression | Split 2 | 0.7484 | 0.8310 | 0.8973 |
| Logistic Regression | Split 3 CV | 0.9559 | 0.9704 | 0.9871 |
| Random Forest | Split 1 | 0.9416 | 0.9617 | 0.9858 |
| Random Forest | Split 2 | 0.7436 | 0.8287 | 0.9160 |
| Random Forest | Split 3 CV | 0.9329 | 0.9562 | 0.9833 |
| MLP Neural Network | Split 1 | 0.9617 | 0.9741 | 0.9937 |
| MLP Neural Network | Split 2 | 0.7837 | 0.8512 | 0.9032 |
| MLP Neural Network | Split 3 CV | 0.9680 | 0.9785 | 0.9928 |
| **SVM (RBF)** | Split 1 | **0.9713** | **0.9805** | **0.9952** |
| SVM (RBF) | Split 2 | 0.7949 | 0.8562 | 0.9161 |
| SVM (RBF) | Split 3 CV | 0.9682 | 0.9784 | 0.9952 |
| KNN (K=9) | Split 1 | 0.9617 | 0.9745 | 0.9809 |
| KNN (K=9) | Split 2 | 0.7676 | 0.8419 | 0.8741 |
| KNN (K=9) | Split 3 CV | 0.9515 | 0.9677 | 0.9838 |
| CNN (thresh=0.7) | Split 2 | 0.8574 | 0.8947 | 0.9499 |

### 3-Class Classification (Normal / Bacteria / Virus)

| Model | Split | Accuracy | Macro F1 |
|-------|-------|----------|----------|
| SVM (OvR) | Split 1 | 0.8151 | 0.8108 |
| SVM (OvR) | Split 2 | 0.6939 | 0.6734 |
| SVM (OvR) | Split 3 CV | 0.7991 | 0.7938 |
| KNN (K=18) | Split 1 | 0.7835 | 0.7656 |
| KNN (K=18) | Split 2 | 0.7019 | 0.6789 |
| KNN (K=18) | Split 3 CV | 0.7697 | 0.7469 |
| CNN (softmax) | Split 2 | 0.7228 | 0.7015 |

---

## Data Splits

- **Split 1** — 80/20 train/test from Dataset 1 (stratified)
- **Split 2** — Dataset 1 train, Dataset 2 val, Dataset 3 test (official)
- **Split 3** — 5-fold StratifiedKFold on Dataset 1

---

## Run Scripts

```bash
# Train all binary models (SVM, KNN, CNN)
/opt/homebrew/bin/python3.11 run_models.py

# Train 3-class models (Normal / Bacteria / Virus)
/opt/homebrew/bin/python3.11 run_3class.py

# Regenerate PDF report
/opt/homebrew/bin/python3.11 make_report.py
```

---

## Use Best Model on a New Image

```python
import joblib
import numpy as np
from PIL import Image

pipeline = joblib.load('models/best_pipeline.joblib')

img = Image.open('new_xray.jpg').convert('L').resize((128, 128))
X = np.array(img).flatten() / 255.0

result = pipeline.predict(X.reshape(1, -1))[0]
prob   = pipeline.predict_proba(X.reshape(1, -1))[0]

print("PNEUMONIA" if result == 1 else "NORMAL")
print(f"Confidence: {max(prob):.2%}")
```

---

## Deliverables

- `notebooks/zoidberg2_pneumonia.ipynb` — full notebook, all models, all splits
- `outputs/zoidberg2_pneumonia_full.html` — exported HTML
- `outputs/synthesis_report.pdf` — written report
- `models/` — all saved model files
- `outputs/figures/` — all plots and confusion matrices
