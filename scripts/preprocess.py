import os
from pathlib import Path
import numpy as np
from PIL import Image
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IMG_SIZE = 128
CACHE = Path('data/preprocessed')
CACHE.mkdir(parents=True, exist_ok=True)

def load_folder(folder):
    X, y = [], []
    for label, val in [('NORMAL', 0), ('PNEUMONIA', 1)]:
        path = Path(folder) / label
        if not path.exists():
            continue
        files = [f for f in sorted(os.listdir(path)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for fname in files:
            try:
                img = Image.open(path / fname).convert('L')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                arr = np.array(img).astype('float32') / 255.0
                X.append(arr.ravel())
                y.append(val)
            except Exception as e:
                print(f"Skipping {path/fname}: {e}")
    return np.array(X), np.array(y)

def build_and_save():
    print('Loading images...')
    X_train, y_train = load_folder('data/train')
    X_val,   y_val   = load_folder('data/val')
    X_test,  y_test  = load_folder('data/test')

    print(f'Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}')

    print('Fitting scaler...')
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    print('Fitting PCA (100 components)...')
    pca = PCA(n_components=100, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)

    # transform val/test
    X_val_pca = pca.transform(scaler.transform(X_val)) if X_val.size else np.empty((0, 100))
    X_test_pca = pca.transform(scaler.transform(X_test)) if X_test.size else np.empty((0, 100))

    # Save raw flattened arrays and labels
    np.save(CACHE / 'X_train.npy', X_train)
    np.save(CACHE / 'y_train.npy', y_train)
    np.save(CACHE / 'X_val.npy', X_val)
    np.save(CACHE / 'y_val.npy', y_val)
    np.save(CACHE / 'X_test.npy', X_test)
    np.save(CACHE / 'y_test.npy', y_test)

    # Save scaler and pca
    joblib.dump(scaler, CACHE / 'scaler.joblib')
    joblib.dump(pca,    CACHE / 'pca.joblib')

    print('Saved preprocessed data to', CACHE)

if __name__ == '__main__':
    build_and_save()
