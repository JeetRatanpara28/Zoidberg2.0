"""
Zoidberg 2.0 — 3-Class Prediction
Classes: 0=NORMAL  1=BACTERIA  2=VIRUS
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

from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

FIGS   = 'outputs/figures'
MODELS = 'models'
os.makedirs(FIGS, exist_ok=True)

CLASS_NAMES = ['NORMAL', 'BACTERIA', 'VIRUS']

# ══════════════════════════════════════════════════════
# 1. LOAD IMAGES WITH 3 LABELS
# ══════════════════════════════════════════════════════
print("="*60)
print("3-CLASS DATA LOADING")
print("="*60)

def load_3class(folder, size=(128, 128)):
    X, y = [], []
    # NORMAL → label 0
    normal_path = os.path.join(folder, 'NORMAL')
    for fname in os.listdir(normal_path):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                img = Image.open(os.path.join(normal_path, fname)).convert('L').resize(size)
                X.append(np.array(img).flatten() / 255.0)
                y.append(0)
            except: pass

    # PNEUMONIA → split by filename
    pneumonia_path = os.path.join(folder, 'PNEUMONIA')
    for fname in os.listdir(pneumonia_path):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                img = Image.open(os.path.join(pneumonia_path, fname)).convert('L').resize(size)
                X.append(np.array(img).flatten() / 255.0)
                # bacteria→1, virus→2
                y.append(1 if 'bacteria' in fname.lower() else 2)
            except: pass

    return np.array(X), np.array(y)

print("Loading train...")
X_train_raw, y_train = load_3class('data/train')
print(f"  NORMAL  : {(y_train==0).sum()}")
print(f"  BACTERIA: {(y_train==1).sum()}")
print(f"  VIRUS   : {(y_train==2).sum()}")
print(f"  Total   : {len(y_train)}")

print("\nLoading test...")
X_test_raw, y_test = load_3class('data/test')
print(f"  NORMAL  : {(y_test==0).sum()}")
print(f"  BACTERIA: {(y_test==1).sum()}")
print(f"  VIRUS   : {(y_test==2).sum()}")
print(f"  Total   : {len(y_test)}")

# ══════════════════════════════════════════════════════
# 2. PREPROCESSING (Scaler + PCA — fit fresh for 3-class)
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("PREPROCESSING (Scaler + PCA)")
print("="*60)

scaler3 = StandardScaler()
X_train_scaled = scaler3.fit_transform(X_train_raw)
X_test_scaled  = scaler3.transform(X_test_raw)

pca3 = PCA(n_components=100, random_state=42)
X_train_pca = pca3.fit_transform(X_train_scaled)
X_test_pca  = pca3.transform(X_test_scaled)

print(f"X_train_pca : {X_train_pca.shape}")
print(f"X_test_pca  : {X_test_pca.shape}")
print(f"Variance explained: {pca3.explained_variance_ratio_.sum():.2%}")

# Train/test split for split1
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_pca, y_train, test_size=0.2, random_state=42, stratify=y_train)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── helper ─────────────────────────────────────────────
def show_metrics(name, y_true, y_pred):
    print(f"\n{'─'*50}")
    print(f"{name}")
    print(f"{'─'*50}")
    print(f"Overall Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1         : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1      : {f1_score(y_true, y_pred, average='weighted'):.4f}")
    print()
    print(classification_report(y_true, y_pred,
          target_names=CLASS_NAMES, digits=4))

def save_cm(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved → {fname}")

# ══════════════════════════════════════════════════════
# 3. SVM — 3-CLASS
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("SVM — 3-CLASS (OvR)")
print("="*60)

# Split 1
svm3_s1 = SVC(kernel='rbf', C=10, gamma='scale',
               class_weight='balanced', probability=True,
               decision_function_shape='ovr', random_state=42)
svm3_s1.fit(X_tr, y_tr)
yp_svm1 = svm3_s1.predict(X_te)
show_metrics("SVM 3-Class — Split 1 (Train/Test)", y_te, yp_svm1)
save_cm(y_te, yp_svm1,
        "SVM 3-Class — Split 1",
        f"{FIGS}/SVM3_split1_cm.png")

# Split 2 (official test)
svm3_s2 = SVC(kernel='rbf', C=10, gamma='scale',
               class_weight='balanced', probability=True,
               decision_function_shape='ovr', random_state=42)
svm3_s2.fit(X_train_pca, y_train)
yp_svm2 = svm3_s2.predict(X_test_pca)
show_metrics("SVM 3-Class — Split 2 (Official Test)", y_test, yp_svm2)
save_cm(y_test, yp_svm2,
        "SVM 3-Class — Split 2 (Official Test)",
        f"{FIGS}/SVM3_split2_cm.png")

# Split 3 — CV
svm3_cv = SVC(kernel='rbf', C=10, gamma='scale',
               class_weight='balanced', probability=True,
               decision_function_shape='ovr', random_state=42)
cv_svm3 = cross_validate(svm3_cv, X_train_pca, y_train, cv=skf,
    scoring=['accuracy','f1_macro','f1_weighted'], n_jobs=-1)
print(f"\nSVM 3-Class — Split 3 (5-Fold CV)")
print(f"  Accuracy     : {cv_svm3['test_accuracy'].mean():.4f} ± {cv_svm3['test_accuracy'].std():.4f}")
print(f"  Macro F1     : {cv_svm3['test_f1_macro'].mean():.4f} ± {cv_svm3['test_f1_macro'].std():.4f}")
print(f"  Weighted F1  : {cv_svm3['test_f1_weighted'].mean():.4f} ± {cv_svm3['test_f1_weighted'].std():.4f}")

# ══════════════════════════════════════════════════════
# 4. KNN — 3-CLASS
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("KNN — 3-CLASS")
print("="*60)

# Find best K
k_scores = []
for k in range(1, 21):
    knn_tmp = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    knn_tmp.fit(X_tr, y_tr)
    k_scores.append(accuracy_score(y_te, knn_tmp.predict(X_te)))

best_k3 = list(range(1,21))[k_scores.index(max(k_scores))]
print(f"\nBest K = {best_k3}  (accuracy = {max(k_scores):.4f})")

# K accuracy plot
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1,21), k_scores, marker='o', color='#0369A1', linewidth=2)
ax.axvline(x=best_k3, color='red', linestyle='--', label=f'Best K={best_k3}')
ax.set_xlabel('K'); ax.set_ylabel('Accuracy')
ax.set_title('KNN 3-Class — Accuracy vs K'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/KNN3_accuracy_vs_k.png', dpi=150)
plt.close()
print(f"  K-plot saved → {FIGS}/KNN3_accuracy_vs_k.png")

# Split 1
knn3_s1 = KNeighborsClassifier(n_neighbors=best_k3, metric='euclidean', n_jobs=-1)
knn3_s1.fit(X_tr, y_tr)
yp_knn1 = knn3_s1.predict(X_te)
show_metrics(f"KNN 3-Class — Split 1 (K={best_k3})", y_te, yp_knn1)
save_cm(y_te, yp_knn1,
        f"KNN 3-Class — Split 1 (K={best_k3})",
        f"{FIGS}/KNN3_split1_cm.png")

# Split 2 (official test)
knn3_s2 = KNeighborsClassifier(n_neighbors=best_k3, metric='euclidean', n_jobs=-1)
knn3_s2.fit(X_train_pca, y_train)
yp_knn2 = knn3_s2.predict(X_test_pca)
show_metrics(f"KNN 3-Class — Split 2 (Official Test, K={best_k3})", y_test, yp_knn2)
save_cm(y_test, yp_knn2,
        f"KNN 3-Class — Split 2 (K={best_k3})",
        f"{FIGS}/KNN3_split2_cm.png")

# Split 3 — CV
knn3_cv = KNeighborsClassifier(n_neighbors=best_k3, metric='euclidean', n_jobs=-1)
cv_knn3 = cross_validate(knn3_cv, X_train_pca, y_train, cv=skf,
    scoring=['accuracy','f1_macro','f1_weighted'], n_jobs=-1)
print(f"\nKNN 3-Class — Split 3 (5-Fold CV)")
print(f"  Accuracy     : {cv_knn3['test_accuracy'].mean():.4f} ± {cv_knn3['test_accuracy'].std():.4f}")
print(f"  Macro F1     : {cv_knn3['test_f1_macro'].mean():.4f} ± {cv_knn3['test_f1_macro'].std():.4f}")
print(f"  Weighted F1  : {cv_knn3['test_f1_weighted'].mean():.4f} ± {cv_knn3['test_f1_weighted'].std():.4f}")

# ══════════════════════════════════════════════════════
# 5. CNN — 3-CLASS
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("CNN — 3-CLASS")
print("="*60)

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 64

def load_3class_images(folder, size=(IMG_SIZE, IMG_SIZE)):
    X, y = [], []
    normal_path = os.path.join(folder, 'NORMAL')
    for fname in os.listdir(normal_path):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                img = Image.open(os.path.join(normal_path, fname)).convert('L').resize(size)
                X.append(np.array(img) / 255.0)
                y.append(0)
            except: pass

    pneumonia_path = os.path.join(folder, 'PNEUMONIA')
    for fname in os.listdir(pneumonia_path):
        if fname.lower().endswith(('.jpg','.jpeg','.png')):
            try:
                img = Image.open(os.path.join(pneumonia_path, fname)).convert('L').resize(size)
                X.append(np.array(img) / 255.0)
                y.append(1 if 'bacteria' in fname.lower() else 2)
            except: pass

    return np.array(X)[..., np.newaxis], np.array(y)

print("Loading images for CNN...")
X_cnn_train, y_cnn_train = load_3class_images('data/train')
X_cnn_test,  y_cnn_test  = load_3class_images('data/test')
# Use part of train as val
X_cnn_tr, X_cnn_val, y_cnn_tr, y_cnn_val = train_test_split(
    X_cnn_train, y_cnn_train, test_size=0.1, random_state=42, stratify=y_cnn_train)

print(f"  CNN Train : {X_cnn_tr.shape} | N:{(y_cnn_tr==0).sum()} B:{(y_cnn_tr==1).sum()} V:{(y_cnn_tr==2).sum()}")
print(f"  CNN Val   : {X_cnn_val.shape}")
print(f"  CNN Test  : {X_cnn_test.shape} | N:{(y_cnn_test==0).sum()} B:{(y_cnn_test==1).sum()} V:{(y_cnn_test==2).sum()}")

# Class weights
cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_cnn_tr)
class_weights = {0: cw[0], 1: cw[1], 2: cw[2]}
print(f"  Class weights → Normal:{cw[0]:.2f}  Bacteria:{cw[1]:.2f}  Virus:{cw[2]:.2f}")

# Build CNN with 3-class output (softmax)
cnn3 = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.4),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax'),   # 3 classes
])

cnn3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',   # multi-class
    metrics=['accuracy']
)
cnn3.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5,
                           restore_best_weights=True, verbose=1)

print("\nTraining CNN 3-class...")
history3 = cnn3.fit(
    X_cnn_tr, y_cnn_tr,
    epochs=25, batch_size=32,
    validation_data=(X_cnn_val, y_cnn_val),
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_cnn3_prob = cnn3.predict(X_cnn_test, verbose=0)
y_cnn3_pred = np.argmax(y_cnn3_prob, axis=1)
show_metrics("CNN 3-Class — Split 2 (Official Test)", y_cnn_test, y_cnn3_pred)
save_cm(y_cnn_test, y_cnn3_pred,
        "CNN 3-Class — Official Test Set",
        f"{FIGS}/CNN3_cm.png")

# Per-class accuracy
cm3 = confusion_matrix(y_cnn_test, y_cnn3_pred)
print("\nPer-class accuracy:")
for i, cls in enumerate(CLASS_NAMES):
    correct = cm3[i, i]
    total   = cm3[i].sum()
    print(f"  {cls:10s}: {correct}/{total}  ({correct/total:.2%})")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history3.history['accuracy'],     label='Train', color='#0369A1')
axes[0].plot(history3.history['val_accuracy'], label='Val',   color='#0D9488')
axes[0].set_title('CNN 3-Class — Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history3.history['loss'],     label='Train', color='#EF4444')
axes[1].plot(history3.history['val_loss'], label='Val',   color='#F97316')
axes[1].set_title('CNN 3-Class — Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIGS}/CNN3_training_curves.png', dpi=150)
plt.close()
print(f"  Training curves → {FIGS}/CNN3_training_curves.png")

# ══════════════════════════════════════════════════════
# 6. FINAL 3-CLASS COMPARISON TABLE
# ══════════════════════════════════════════════════════
print("\n" + "="*60)
print("3-CLASS FINAL COMPARISON")
print("="*60)

print(f"\n{'Model':<30} {'Split':<12} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}")
print("-"*78)

# SVM
print(f"{'SVM':<30} {'Split 1':>12} {accuracy_score(y_te, yp_svm1):>10.4f} {f1_score(y_te, yp_svm1, average='macro'):>10.4f} {f1_score(y_te, yp_svm1, average='weighted'):>12.4f}")
print(f"{'SVM':<30} {'Split 2':>12} {accuracy_score(y_test, yp_svm2):>10.4f} {f1_score(y_test, yp_svm2, average='macro'):>10.4f} {f1_score(y_test, yp_svm2, average='weighted'):>12.4f}")
print(f"{'SVM':<30} {'Split 3 CV':>12} {cv_svm3['test_accuracy'].mean():>10.4f} {cv_svm3['test_f1_macro'].mean():>10.4f} {cv_svm3['test_f1_weighted'].mean():>12.4f}")

# KNN
print(f"{'KNN (K='+str(best_k3)+')':<30} {'Split 1':>12} {accuracy_score(y_te, yp_knn1):>10.4f} {f1_score(y_te, yp_knn1, average='macro'):>10.4f} {f1_score(y_te, yp_knn1, average='weighted'):>12.4f}")
print(f"{'KNN (K='+str(best_k3)+')':<30} {'Split 2':>12} {accuracy_score(y_test, yp_knn2):>10.4f} {f1_score(y_test, yp_knn2, average='macro'):>10.4f} {f1_score(y_test, yp_knn2, average='weighted'):>12.4f}")
print(f"{'KNN (K='+str(best_k3)+')':<30} {'Split 3 CV':>12} {cv_knn3['test_accuracy'].mean():>10.4f} {cv_knn3['test_f1_macro'].mean():>10.4f} {cv_knn3['test_f1_weighted'].mean():>12.4f}")

# CNN
print(f"{'CNN':<30} {'Split 2':>12} {accuracy_score(y_cnn_test, y_cnn3_pred):>10.4f} {f1_score(y_cnn_test, y_cnn3_pred, average='macro'):>10.4f} {f1_score(y_cnn_test, y_cnn3_pred, average='weighted'):>12.4f}")

# ── Comparison bar chart ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models_3  = ['SVM\nSplit1', 'SVM\nSplit2', 'SVM\nCV', 'KNN\nSplit1', 'KNN\nSplit2', 'KNN\nCV', 'CNN\nSplit2']
acc_3     = [
    accuracy_score(y_te, yp_svm1), accuracy_score(y_test, yp_svm2), cv_svm3['test_accuracy'].mean(),
    accuracy_score(y_te, yp_knn1), accuracy_score(y_test, yp_knn2), cv_knn3['test_accuracy'].mean(),
    accuracy_score(y_cnn_test, y_cnn3_pred)
]
f1_3      = [
    f1_score(y_te, yp_svm1, average='macro'), f1_score(y_test, yp_svm2, average='macro'), cv_svm3['test_f1_macro'].mean(),
    f1_score(y_te, yp_knn1, average='macro'), f1_score(y_test, yp_knn2, average='macro'), cv_knn3['test_f1_macro'].mean(),
    f1_score(y_cnn_test, y_cnn3_pred, average='macro')
]
colors_bar = ['#0369A1']*3 + ['#0D9488']*3 + ['#7C3AED']

axes[0].bar(models_3, acc_3, color=colors_bar)
axes[0].set_ylim(0, 1); axes[0].set_ylabel('Accuracy')
axes[0].set_title('3-Class — Accuracy by Model & Split')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(acc_3):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

axes[1].bar(models_3, f1_3, color=colors_bar)
axes[1].set_ylim(0, 1); axes[1].set_ylabel('Macro F1')
axes[1].set_title('3-Class — Macro F1 by Model & Split')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(f1_3):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIGS}/3class_comparison.png', dpi=150)
plt.close()
print(f"\nComparison chart → {FIGS}/3class_comparison.png")

# Save models
joblib.dump(svm3_s2,  f'{MODELS}/svm3_best.joblib')
joblib.dump(knn3_s2,  f'{MODELS}/knn3_best.joblib')
cnn3.save(f'{MODELS}/cnn3_best.keras')
print(f"Models saved: svm3_best.joblib, knn3_best.joblib, cnn3_best.keras")

print("\n" + "="*60)
print("3-CLASS PREDICTION COMPLETE!")
print("="*60)
