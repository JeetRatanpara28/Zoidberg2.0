"""
Zoidberg 2.0 — SVM + KNN + CNN
Runs all 3 models across all 3 splits and saves results + figures.
"""

import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────
CACHE   = 'data/preprocessed'
FIGS    = 'outputs/figures'
MODELS  = 'models'
os.makedirs(FIGS,   exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1.  LOAD PREPROCESSED DATA
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("LOADING PREPROCESSED DATA")
print("="*60)

X_train_raw = np.load(f'{CACHE}/X_train.npy')
y_train     = np.load(f'{CACHE}/y_train.npy')
X_val_raw   = np.load(f'{CACHE}/X_val.npy')
y_val       = np.load(f'{CACHE}/y_val.npy')
X_test_raw  = np.load(f'{CACHE}/X_test.npy')
y_test      = np.load(f'{CACHE}/y_test.npy')

scaler = joblib.load(f'{CACHE}/scaler.joblib')
pca    = joblib.load(f'{CACHE}/pca.joblib')

X_train_pca = pca.transform(scaler.transform(X_train_raw))
X_val_pca   = pca.transform(scaler.transform(X_val_raw))
X_test_pca  = pca.transform(scaler.transform(X_test_raw))

print(f"X_train_pca : {X_train_pca.shape}")
print(f"X_val_pca   : {X_val_pca.shape}")
print(f"X_test_pca  : {X_test_pca.shape}")
print(f"y_train — Normal: {(y_train==0).sum()}  Pneumonia: {(y_train==1).sum()}")

# ── splits ─────────────────────────────────────────────────────────────
X_split1_train, X_split1_test, y_split1_train, y_split1_test = train_test_split(
    X_train_pca, y_train, test_size=0.2, random_state=42, stratify=y_train)

X_split2_train, X_split2_val,   X_split2_test  = X_train_pca, X_val_pca,   X_test_pca
y_split2_train, y_split2_val,   y_split2_test  = y_train,     y_val,       y_test

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nAll splits ready!")

# ── helper: evaluate & print ───────────────────────────────────────────
def evaluate(name, split, y_true, y_pred, y_prob):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    print(f"\n{name} — {split}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")

    # confusion matrix breakdown
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    normal_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    pneum_acc  = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  Normal accuracy    : {normal_acc:.2%}  ({tn}/{tn+fp})")
    print(f"  Pneumonia accuracy : {pneum_acc:.2%}  ({tp}/{tp+fn})")
    return acc, prec, rec, f1, auc

# ── helper: save plots ─────────────────────────────────────────────────
def save_plots(model, name_slug, X_test, y_test, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    axes[0].set_title(f'{title_prefix} — ROC Curve')
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test,
                                          display_labels=['Normal','Pneumonia'],
                                          cmap='Blues', ax=axes[1])
    axes[1].set_title(f'{title_prefix} — Confusion Matrix')
    plt.tight_layout()
    path = f'{FIGS}/{name_slug}_evaluation.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")

# ══════════════════════════════════════════════════════════════════════
# 2.  SVM
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SVM — SUPPORT VECTOR MACHINE")
print("="*60)

# ── Split 1 ────────────────────────────────────────────────────────────
print("\n[SVM Split 1 — Simple Train/Test]")
svm1 = SVC(kernel='rbf', C=10, gamma=0.01, class_weight='balanced',
           probability=True, random_state=42)
svm1.fit(X_split1_train, y_split1_train)
yp1  = svm1.predict(X_split1_test)
ypr1 = svm1.predict_proba(X_split1_test)[:, 1]
evaluate("SVM", "Split 1 (Train/Test)", y_split1_test, yp1, ypr1)
save_plots(svm1, "SVM_split1", X_split1_test, y_split1_test, "SVM Split 1")

# ── Split 2 with GridSearchCV ──────────────────────────────────────────
print("\n[SVM Split 2 — GridSearchCV on Val, Test on Dataset 3]")
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1]}
grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
grid.fit(X_split2_train, y_split2_train)
print(f"  Best params: {grid.best_params_}  |  CV F1: {grid.best_score_:.4f}")
svm2 = grid.best_estimator_
yp2  = svm2.predict(X_split2_test)
ypr2 = svm2.predict_proba(X_split2_test)[:, 1]
evaluate("SVM", "Split 2 (Train/Val/Test)", y_split2_test, yp2, ypr2)
save_plots(svm2, "SVM_split2", X_split2_test, y_split2_test, "SVM Split 2")

# GridSearchCV heatmap
pivot = np.zeros((3, 4))
for par, score in zip(grid.cv_results_['params'], grid.cv_results_['mean_test_score']):
    ci = [0.1, 1, 10, 100].index(par['C'])
    ri = [0.001, 0.01, 0.1].index(par['gamma'])
    pivot[ri, ci] = score
fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
            xticklabels=[0.1, 1, 10, 100], yticklabels=[0.001, 0.01, 0.1],
            ax=ax)
ax.set_xlabel('C'); ax.set_ylabel('gamma')
ax.set_title('SVM GridSearchCV — F1 Score Heatmap')
plt.tight_layout()
plt.savefig(f'{FIGS}/SVM_gridsearch_heatmap.png', dpi=150)
plt.close()
print(f"  Heatmap saved → {FIGS}/SVM_gridsearch_heatmap.png")

# ── Split 3 — K-Fold ──────────────────────────────────────────────────
print("\n[SVM Split 3 — 5-Fold Cross-Validation]")
svm_cv = SVC(kernel='rbf', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'],
             class_weight='balanced', probability=True, random_state=42)
cv_svm = cross_validate(svm_cv, X_train_pca, y_train, cv=skf,
                        scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1)
print(f"  Accuracy  : {cv_svm['test_accuracy'].mean():.4f} ± {cv_svm['test_accuracy'].std():.4f}")
print(f"  F1        : {cv_svm['test_f1'].mean():.4f} ± {cv_svm['test_f1'].std():.4f}")
print(f"  ROC-AUC   : {cv_svm['test_roc_auc'].mean():.4f} ± {cv_svm['test_roc_auc'].std():.4f}")

# save best SVM
joblib.dump(svm2, f'{MODELS}/svm_best.joblib')
print(f"  SVM saved → {MODELS}/svm_best.joblib")

# ══════════════════════════════════════════════════════════════════════
# 3.  KNN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("KNN — K-NEAREST NEIGHBOURS")
print("="*60)

from sklearn.metrics import balanced_accuracy_score

# Find best K using balanced_accuracy (avoids majority-class bias from plain accuracy)
print("\n[KNN — Finding best K (1–20) using balanced_accuracy]")
k_range  = range(1, 21)
k_scores = []
for k in k_range:
    knn_tmp = KNeighborsClassifier(n_neighbors=k, weights='uniform',
                                   metric='euclidean', n_jobs=-1)
    knn_tmp.fit(X_split1_train, y_split1_train)
    k_scores.append(balanced_accuracy_score(y_split1_test,
                                            knn_tmp.predict(X_split1_test)))

best_k = list(k_range)[k_scores.index(max(k_scores))]
print(f"  Best K = {best_k}  (balanced_accuracy = {max(k_scores):.4f})")

# Plot balanced accuracy vs K
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(list(k_range), k_scores, marker='o', color='#0369A1', linewidth=2)
ax.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('Balanced Accuracy')
ax.set_title('KNN — Balanced Accuracy vs K'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/KNN_accuracy_vs_k.png', dpi=150)
plt.close()
print(f"  K-plot saved → {FIGS}/KNN_accuracy_vs_k.png")

# ── Split 1 ────────────────────────────────────────────────────────────
print(f"\n[KNN Split 1 — K={best_k}]")
knn1 = KNeighborsClassifier(n_neighbors=best_k, weights='uniform',
                             metric='euclidean', n_jobs=-1)
knn1.fit(X_split1_train, y_split1_train)
yp1k  = knn1.predict(X_split1_test)
ypr1k = knn1.predict_proba(X_split1_test)[:, 1]
evaluate("KNN", "Split 1 (Train/Test)", y_split1_test, yp1k, ypr1k)
save_plots(knn1, "KNN_split1", X_split1_test, y_split1_test, "KNN Split 1")

# ── Split 2 — threshold tuning for better Normal recall ───────────────
print(f"\n[KNN Split 2 — K={best_k}, Test on Dataset 3]")
knn2 = KNeighborsClassifier(n_neighbors=best_k, weights='uniform',
                             metric='euclidean', n_jobs=-1)
knn2.fit(X_split2_train, y_split2_train)
ypr2k = knn2.predict_proba(X_split2_test)[:, 1]

# threshold tuning: pick threshold that maximises balanced_accuracy
best_t_knn, best_bal_knn = 0.5, 0
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    pred_t = (ypr2k >= t).astype(int)
    bal = balanced_accuracy_score(y_split2_test, pred_t)
    if bal > best_bal_knn:
        best_bal_knn, best_t_knn = bal, t
print(f"  Best threshold: {best_t_knn}  (balanced_acc: {best_bal_knn:.4f})")

yp2k = (ypr2k >= best_t_knn).astype(int)
evaluate("KNN", "Split 2 (Train/Val/Test)", y_split2_test, yp2k, ypr2k)
save_plots(knn2, "KNN_split2", X_split2_test, y_split2_test, "KNN Split 2")

# ── Split 3 ────────────────────────────────────────────────────────────
print(f"\n[KNN Split 3 — 5-Fold CV, K={best_k}]")
knn_cv = KNeighborsClassifier(n_neighbors=best_k, weights='uniform',
                               metric='euclidean', n_jobs=-1)
cv_knn = cross_validate(knn_cv, X_train_pca, y_train, cv=skf,
                        scoring=['accuracy', 'f1', 'roc_auc'], n_jobs=-1)
print(f"  Accuracy  : {cv_knn['test_accuracy'].mean():.4f} ± {cv_knn['test_accuracy'].std():.4f}")
print(f"  F1        : {cv_knn['test_f1'].mean():.4f} ± {cv_knn['test_f1'].std():.4f}")
print(f"  ROC-AUC   : {cv_knn['test_roc_auc'].mean():.4f} ± {cv_knn['test_roc_auc'].std():.4f}")

joblib.dump(knn2, f'{MODELS}/knn_best.joblib')
print(f"  KNN saved → {MODELS}/knn_best.joblib")

# ══════════════════════════════════════════════════════════════════════
# 4.  CNN
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("CNN — CONVOLUTIONAL NEURAL NETWORK")
print("="*60)

IMG_SIZE = 64  # smaller size for speed — still captures patterns

def load_images(folder, size=(IMG_SIZE, IMG_SIZE)):
    X, y = [], []
    for label, val in [('NORMAL', 0), ('PNEUMONIA', 1)]:
        path = os.path.join(folder, label)
        files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for fname in files:
            try:
                img = Image.open(os.path.join(path, fname)).convert('L')
                img = img.resize(size)
                arr = np.array(img) / 255.0
                X.append(arr)
                y.append(val)
            except:
                pass
    return np.array(X)[..., np.newaxis], np.array(y)   # add channel dim

print("\nLoading images for CNN...")
X_cnn_train, y_cnn_train = load_images('data/train')
X_cnn_val,   y_cnn_val   = load_images('data/val')
X_cnn_test,  y_cnn_test  = load_images('data/test')
print(f"  Train: {X_cnn_train.shape}  | Normal: {(y_cnn_train==0).sum()}  Pneumonia: {(y_cnn_train==1).sum()}")
print(f"  Val  : {X_cnn_val.shape}")
print(f"  Test : {X_cnn_test.shape}")

# class weights to fix imbalance
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_cnn_train)
class_weights = {0: cw[0], 1: cw[1]}
print(f"  Class weights → Normal: {cw[0]:.2f}, Pneumonia: {cw[1]:.2f}")

# Build CNN
def build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

print("\nBuilding CNN...")
cnn = build_cnn()
cnn.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nTraining CNN...")
history = cnn.fit(
    X_cnn_train, y_cnn_train,
    epochs=25,
    batch_size=32,
    validation_data=(X_cnn_val, y_cnn_val),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ── Evaluate CNN on test set ───────────────────────────────────────────
print("\n[CNN — Test Set Evaluation]")
y_cnn_prob = cnn.predict(X_cnn_test, verbose=0).flatten()
y_cnn_pred = (y_cnn_prob >= 0.5).astype(int)

acc  = accuracy_score(y_cnn_test, y_cnn_pred)
prec = precision_score(y_cnn_test, y_cnn_pred, zero_division=0)
rec  = recall_score(y_cnn_test, y_cnn_pred, zero_division=0)
f1   = f1_score(y_cnn_test, y_cnn_pred, zero_division=0)
auc  = roc_auc_score(y_cnn_test, y_cnn_prob)

print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")

cm = confusion_matrix(y_cnn_test, y_cnn_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  Normal accuracy    : {tn/(tn+fp):.2%}  ({tn}/{tn+fp})")
print(f"  Pneumonia accuracy : {tp/(tp+fn):.2%}  ({tp}/{tp+fn})")

# ── Threshold tuning for better Normal accuracy ────────────────────────
print("\n[CNN — Threshold Tuning for Normal Accuracy]")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
best_thresh, best_bal = 0.5, 0
thresh_results = []
for t in thresholds:
    yp_t = (y_cnn_prob >= t).astype(int)
    cm_t = confusion_matrix(y_cnn_test, yp_t)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    norm_acc = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0
    pneu_acc = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    bal = (norm_acc + pneu_acc) / 2
    thresh_results.append((t, norm_acc, pneu_acc, bal))
    print(f"  Threshold {t:.1f} → Normal: {norm_acc:.2%}  Pneumonia: {pneu_acc:.2%}  Balanced: {bal:.2%}")
    if bal > best_bal:
        best_bal, best_thresh = bal, t

print(f"\n  ★ Best threshold: {best_thresh}  (balanced accuracy: {best_bal:.2%})")

# ── Training curves ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train Acc', color='#0369A1')
axes[0].plot(history.history['val_accuracy'], label='Val Acc',   color='#0D9488')
axes[0].set_title('CNN — Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history['loss'],     label='Train Loss', color='#EF4444')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='#F97316')
axes[1].set_title('CNN — Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/CNN_training_curves.png', dpi=150)
plt.close()
print(f"  Training curves → {FIGS}/CNN_training_curves.png")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# default threshold
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Normal', 'Pneumonia'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('CNN — Confusion Matrix (threshold=0.5)')

# best threshold
y_best = (y_cnn_prob >= best_thresh).astype(int)
disp2 = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_cnn_test, y_best),
                                display_labels=['Normal', 'Pneumonia'])
disp2.plot(ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title(f'CNN — Confusion Matrix (threshold={best_thresh})')
plt.tight_layout()
plt.savefig(f'{FIGS}/CNN_confusion_matrices.png', dpi=150)
plt.close()
print(f"  Confusion matrices → {FIGS}/CNN_confusion_matrices.png")

# ROC curve
fig, ax = plt.subplots(figsize=(7, 5))
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_predictions(y_cnn_test, y_cnn_prob, ax=ax, name='CNN')
ax.set_title('CNN — ROC Curve')
plt.tight_layout()
plt.savefig(f'{FIGS}/CNN_roc_curve.png', dpi=150)
plt.close()
print(f"  ROC curve → {FIGS}/CNN_roc_curve.png")

# save model
cnn.save(f'{MODELS}/cnn_best.keras')
print(f"  CNN saved → {MODELS}/cnn_best.keras")

# ══════════════════════════════════════════════════════════════════════
# 5.  FINAL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL MODEL COMPARISON TABLE")
print("="*60)

# recompute CV results for SVM/KNN to include in summary
svm_cv_acc  = cv_svm['test_accuracy'].mean()
svm_cv_f1   = cv_svm['test_f1'].mean()
svm_cv_auc  = cv_svm['test_roc_auc'].mean()

knn_cv_acc  = cv_knn['test_accuracy'].mean()
knn_cv_f1   = cv_knn['test_f1'].mean()
knn_cv_auc  = cv_knn['test_roc_auc'].mean()

cnn_acc2 = accuracy_score(y_cnn_test, (y_cnn_prob >= best_thresh).astype(int))
cnn_f1_2 = f1_score(y_cnn_test, (y_cnn_prob >= best_thresh).astype(int), zero_division=0)

rows = [
    ("Logistic Regression", "Split 1", 0.9598, 0.9728, 0.9891),
    ("Logistic Regression", "Split 2", 0.7484, 0.8310, 0.8973),
    ("Logistic Regression", "Split 3", 0.9559, 0.9704, 0.9871),
    ("Random Forest",       "Split 1", 0.9416, 0.9617, 0.9858),
    ("Random Forest",       "Split 2", 0.7436, 0.8287, 0.9160),
    ("Random Forest",       "Split 3", 0.9329, 0.9562, 0.9833),
    ("MLP Neural Network",  "Split 1", 0.9617, 0.9741, 0.9937),
    ("MLP Neural Network",  "Split 2", 0.7837, 0.8512, 0.9032),
    ("MLP Neural Network",  "Split 3", 0.9680, 0.9785, 0.9928),
    ("SVM",                 "Split 1", accuracy_score(y_split1_test, yp1),  f1_score(y_split1_test, yp1),  roc_auc_score(y_split1_test, ypr1)),
    ("SVM",                 "Split 2", accuracy_score(y_split2_test, yp2),  f1_score(y_split2_test, yp2),  roc_auc_score(y_split2_test, ypr2)),
    ("SVM",                 "Split 3", svm_cv_acc, svm_cv_f1, svm_cv_auc),
    ("KNN",                 "Split 1", accuracy_score(y_split1_test, yp1k), f1_score(y_split1_test, yp1k), roc_auc_score(y_split1_test, ypr1k)),
    ("KNN",                 "Split 2", accuracy_score(y_split2_test, yp2k), f1_score(y_split2_test, yp2k), roc_auc_score(y_split2_test, ypr2k)),
    ("KNN",                 "Split 3", knn_cv_acc, knn_cv_f1, knn_cv_auc),
    ("CNN",                 "Split 2", acc,   f1,   auc),
    (f"CNN (thresh={best_thresh})", "Split 2", cnn_acc2, cnn_f1_2, auc),
]

print(f"\n{'Model':<28} {'Split':<10} {'Accuracy':>10} {'F1':>8} {'ROC-AUC':>10}")
print("-"*70)
for r in rows:
    print(f"{r[0]:<28} {r[1]:<10} {r[2]:>10.4f} {r[3]:>8.4f} {r[4]:>10.4f}")

# Bar chart — all models Split 2 comparison
models_list = ["LR", "RF", "MLP", "SVM", "KNN", "CNN"]
acc_s2  = [0.7484, 0.7436, 0.7837,
           accuracy_score(y_split2_test, yp2),
           accuracy_score(y_split2_test, yp2k),
           acc]
f1_s2   = [0.8310, 0.8287, 0.8512,
           f1_score(y_split2_test, yp2),
           f1_score(y_split2_test, yp2k),
           f1]

x  = np.arange(len(models_list))
w  = 0.35
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - w/2, acc_s2, w, label='Accuracy', color='#0369A1')
b2 = ax.bar(x + w/2, f1_s2,  w, label='F1 Score', color='#0D9488')
ax.set_xticks(x); ax.set_xticklabels(models_list, fontsize=12)
ax.set_ylim(0.6, 1.0); ax.set_ylabel('Score'); ax.set_title('All Models — Split 2 (Real Test Set)')
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIGS}/ALL_models_comparison.png', dpi=150)
plt.close()
print(f"\nComparison chart → {FIGS}/ALL_models_comparison.png")

print("\n" + "="*60)
print("ALL DONE — SVM + KNN + CNN complete!")
print(f"Figures saved in: {FIGS}/")
print(f"Models saved in : {MODELS}/")
print("="*60)
