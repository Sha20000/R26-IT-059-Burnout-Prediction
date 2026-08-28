import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS

app     = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
META_DIR   = os.path.join(BASE_DIR, 'meta_results')

# ── Load model + config at startup ───────────────────────

class MetaFNN(nn.Module):
    def __init__(self, n=10, h=32, d=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, h), nn.ReLU(), nn.Dropout(d),
            nn.Linear(h, h//2), nn.ReLU(), nn.Dropout(d),
            nn.Linear(h//2, h//4), nn.ReLU(),
            nn.Linear(h//4, 1)
        )
    def forward(self, x):
        return self.net(x)

print("=" * 50)
print("R26-IT-059 | Meta-Integration API")
print("=" * 50)

# Load config
with open(os.path.join(
        META_DIR, 'meta_config.json')) as f:
    config = json.load(f)

THRESHOLD    = config['threshold']
FEATURE_COLS = config['feature_cols']
N_FEATURES   = config['n_features']

# Load model
model = MetaFNN(n=N_FEATURES,h=32, d=0.3)
model.load_state_dict(torch.load(
    os.path.join(META_DIR, 'meta_fnn_best.pt'),
    map_location='cpu', weights_only=True))
model.eval()
print("Meta FNN loaded")

# Load scaler
with open(os.path.join(
        META_DIR, 'meta_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded")

# Load predictions CSV
predictions_df = pd.read_csv(os.path.join(
    META_DIR, 'final_burnout_predictions.csv'))
print(f"Predictions loaded: {len(predictions_df)} students")
print(f"API ready at http://localhost:5002")
print("=" * 50)

# ── Helper ────────────────────────────────────────────────
def safe_row(row):
    d = {}
    for k, v in row.items():
        if pd.isna(v):
            d[k] = None
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = round(float(v), 4)
        else:
            d[k] = v
    return d

# ── Routes ────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'api':     'R26-IT-059 Meta-Integration API',
        'version': '1.0',
        'routes': [
            '/health',
            '/students',
            '/students/<id>',
            '/students?alert=HIGH',
            '/summary',
            '/predict'
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'ok',
        'model':   'Meta-Integration FNN',
        'project': 'R26-IT-059',
        'components': {
            'IT22916426': 'GRU Academic Risk',
            'IT22215710': 'VAE Behavioural Anomaly'
        },
        'f1':      config['results']['f1'],
        'auc':     config['results']['auc'],
        'total':   len(predictions_df)
    })


@app.route('/students', methods=['GET'])
def get_students():
    alert = request.args.get('alert', 'ALL').upper()

    if alert in ['HIGH', 'MEDIUM', 'LOW']:
        df = predictions_df[
            predictions_df['final_alert'] == alert]
    else:
        df = predictions_df

    students = [safe_row(row)
                for _, row in df.iterrows()]
    return jsonify({
        'total':    len(students),
        'filter':   alert,
        'students': students
    })


@app.route('/students/<student_id>', methods=['GET'])
def get_student(student_id):
    match = predictions_df[
        predictions_df['student_id'].astype(str)
        == str(student_id)]

    if match.empty:
        return jsonify(
            {'error': f'{student_id} not found'}), 404

    return jsonify(safe_row(match.iloc[0]))


@app.route('/summary', methods=['GET'])
def summary():
    high   = int((predictions_df[
        'final_alert'] == 'HIGH').sum())
    medium = int((predictions_df[
        'final_alert'] == 'MEDIUM').sum())
    low    = int((predictions_df[
        'final_alert'] == 'LOW').sum())

    return jsonify({
        'total_students': len(predictions_df),
        'high_risk':      high,
        'medium_risk':    medium,
        'low_risk':       low,
        'model_f1':       config['results']['f1'],
        'model_auc':      config['results']['auc'],
        'model_accuracy': config['results']['accuracy'],
        'component_contribution':
            config['component_contribution'],
        'threshold':      THRESHOLD
    })


@app.route('/predict', methods=['POST'])
def predict_live():
    """
    Live prediction endpoint.
    POST JSON with feature values.
    Returns final burnout risk score.
    """
    try:
        data = request.get_json()

        # Build feature vector in correct order
        features = []
        for col in FEATURE_COLS:
            val = data.get(col, 0)
            features.append(float(val))

        X = np.array([features], dtype=np.float32)
        X_scaled = scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            logit = model(X_tensor).squeeze(-1)
            prob  = torch.sigmoid(logit).item()

        alert = ('HIGH'   if prob > 0.70 else
                 'MEDIUM' if prob > 0.40 else 'LOW')

        # Explicit weighted formula (transparent, non-learned)
        # GRU 0.4 + VAE 0.3 + Emotional-NLP 0.3
        academic_risk = float(data.get('academic_risk', 0))
        behavior_risk_mean = float(data.get('behavior_risk_mean', 0))
        emotional_stress_score = float(
            data.get('emotional_stress_score', 0))
        emo_norm = max(0.0, min(1.0, emotional_stress_score / 4.0))

        weighted_risk = round(
            0.4 * academic_risk +
            0.3 * behavior_risk_mean +
            0.3 * emo_norm, 4)
        weighted_alert = (
            'HIGH'   if weighted_risk > 0.70 else
            'MEDIUM' if weighted_risk > 0.40 else 'LOW')

        return jsonify({
            'final_burnout_risk':     round(prob, 4),
            'final_alert':            alert,
            'weighted_burnout_risk':  weighted_risk,
            'weighted_alert':         weighted_alert,
            'threshold_used':         THRESHOLD,
            'features_used':          FEATURE_COLS,
            'source':                 'live_inference'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0', threaded=True)
