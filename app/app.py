from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import joblib
import io
import os
import traceback

app = Flask(__name__)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')
CACHE_DIR = os.path.join(BASE, 'data', 'preprocessed')

# Load scaler and PCA for 3-class models
print("Loading scaler and PCA...")
scaler = joblib.load(os.path.join(CACHE_DIR, 'scaler.joblib'))
pca = joblib.load(os.path.join(CACHE_DIR, 'pca.joblib'))
print("Done!")

# Binary models — already have scaler+PCA inside pipeline
binary_models = {}
for name, fname in [
    ('Best (MLP)', 'best_pipeline.joblib'),
    ('SVM', 'svm_best.joblib'),
    ('KNN', 'knn_best.joblib')
]:
    try:
        binary_models[name] = joblib.load(os.path.join(MODELS_DIR, fname))
        print(f"  {name} loaded")
    except Exception as e:
        print(f"  {name} failed: {e}")

# 3-class models — need manual scaler+PCA
three_models = {}
for name, fname in [
    ('SVM (3-class)', 'svm3_best.joblib'),
    ('KNN (3-class)', 'knn3_best.joblib')
]:
    try:
        three_models[name] = joblib.load(os.path.join(MODELS_DIR, fname))
        print(f"  {name} loaded")
    except Exception as e:
        print(f"  {name} failed: {e}")

# CNN 3-class model
cnn_3class = None
try:
    import tensorflow as tf
    cnn_3class = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'cnn3_best.keras'))
    three_models['CNN (3-class)'] = 'cnn'
    print("  CNN 3-class loaded")
except Exception as e:
    print(f"  CNN 3-class failed: {e}")

print(f"\nReady! {len(binary_models)} binary models, {len(three_models)} 3-class models")


def preprocess_image(file_bytes):
    """For binary models — raw flatten only. Scaler+PCA inside pipeline."""
    img = Image.open(io.BytesIO(file_bytes)).convert('L').resize((128, 128))
    X = np.array(img).flatten() / 255.0
    return X.reshape(1, -1)


def preprocess_image_pca(file_bytes):
    """For 3-class sklearn models — apply scaler+PCA manually."""
    img = Image.open(io.BytesIO(file_bytes)).convert('L').resize((128, 128))
    X = np.array(img).flatten() / 255.0
    X = X.reshape(1, -1)
    return pca.transform(scaler.transform(X))


def preprocess_image_cnn(file_bytes):
    """For CNN — resize to 64x64, reshape to (1, 64, 64, 1)."""
    img = Image.open(io.BytesIO(file_bytes)).convert('L').resize((64, 64))
    X = np.array(img).reshape(1, 64, 64, 1) / 255.0
    return X


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file_bytes = request.files['image'].read()
        model_name = request.form.get('model', 'Best (MLP)')

        # Binary prediction
        if model_name in binary_models:
            model = binary_models[model_name]
            X = preprocess_image(file_bytes)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            prediction = 'PNEUMONIA' if pred == 1 else 'NORMAL'
            probs = {
                'NORMAL': round(proba[0] * 100, 1),
                'PNEUMONIA': round(proba[1] * 100, 1)
            }
            return jsonify({
                'prediction': prediction,
                'probabilities': probs,
                'model': model_name,
                'type': 'binary',
                'classes': ['NORMAL', 'PNEUMONIA']
            })

        # 3-class prediction
        elif model_name in three_models:
            class_names = ['NORMAL', 'BACTERIA', 'VIRUS']

            if model_name == 'CNN (3-class)' and cnn_3class is not None:
                X = preprocess_image_cnn(file_bytes)
                proba = cnn_3class.predict(X)[0]
            else:
                model = three_models[model_name]
                X = preprocess_image_pca(file_bytes)
                proba = model.predict_proba(X)[0]

            pred_idx = int(np.argmax(proba))
            prediction = class_names[pred_idx]
            probs = {
                class_names[i]: round(float(proba[i]) * 100, 1)
                for i in range(3)
            }
            return jsonify({
                'prediction': prediction,
                'probabilities': probs,
                'model': model_name,
                'type': '3class',
                'classes': class_names
            })

        else:
            return jsonify({'error': f'Model not found: {model_name}'}), 400

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("Zoidberg 2.0 — server starting...")
    print("Open browser: http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)