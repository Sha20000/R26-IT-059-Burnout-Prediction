import os;
import sys;
import json;
import pandas as pd;
from flask import Flask, request, jsonify;
from flask_cors import CORS;

app = Flask(__name__);
CORS(app);

#Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__));
RESULTS_PATH = os.path.join(BASE_DIR, '..','results','metrics');
CSV_PATH = os.path.join(RESULTS_PATH, 'model8_predictions.csv');
JSON_PATH = os.path.join(RESULTS_PATH,'model8_unified_gru_results.json');

#Load predictions once at startups  
print("=" * 50)
print("R26-IT-059 | Academic Burnout Detection API")
print("IT22916426 | Mahavitha S.M.")
print("=" * 50)

try:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} student predictions from CSV")
except FileNotFoundError:
    print(f"ERROR: CSV not found at {CSV_PATH}")
    print("Run model_8_unified_gru.py first to generate predictions")
    df = pd.DataFrame()

#Load model result JSOM for summary stats

try:
    with open(JSON_PATH, 'r') as f:
        model_results = json.load(f)
    print(f"Loaded model results JSON")    

except FileNotFoundError:
    model_results = {}
    print("WARNING: model results JSON not found")

print(f"API ready at http://localhost:5001")
print("=" * 50)   

#Helper

def clean_row(row):
    """Convert a DataFrame row to a clean JSON-safe dict."""
    d = {}
    for k,v in row.items():
        if pd.isna(v):
            d[k] = None
        elif isinstance(v,float):
            d[k] = round(float(v),4)
        else:
            d[k] = v

    return d

#Routes

@app.route('/health',methods = ['GET'])
def health_check():
    """Health check — confirms API is running."""
    return jsonify({
        'status': 'ok',
        'model': 'Model 8 — Unified Multi-Horizon GRU',
        'project': 'R26-IT-059',
        'student': 'IT22916426',
        'total_predictions': len(df)
    })

@app.route('/students', methods = ['GET'])
def get_all_students():
    """
    Returns all student predictions.
    Optional filter: ?alert=HIGH or ?alert=MEDIUM or ?alert=LOW
    """

    if df.empty:
        return jsonify({'error': 'No predictions loaded'}), 500
    
    alert_filter = request.args.get('alert','ALL').upper()

    if alert_filter in ['HIGH','MEDIUM','LOW']:
        filtered = df[df['alert_level'] == alert_filter]
    else:
        filtered = df

    students = [clean_row(row) for _, row in filtered.iterrows()]

    return jsonify({
        'total': len(students),
        'filter': alert_filter,
        'students': students
    })  


@app.route('/students/<int:student_idx>', methods = ['GET'])
def get_student(student_idx):
    """
    Returns full prediction for one student by index.
    Example: /students/7
    """
    
    if df.empty:
        return jsonify({'error': 'No predictions loaded'}), 500
    
    if student_idx < 0 or student_idx >= len(df):
        return jsonify({'error': f'Student index {student_idx} out of range (0-{len(df)-1})'}),404
    
    row = df.iloc[student_idx]
    return jsonify(clean_row(row))

@app.route('/summary', methods = ['GET'])    
def get_summary():
    """
    Returns dashboard summary statistics.
    Total students, HIGH/MEDIUM/LOW counts, model performance.
    """
     
    if df.empty:
           return jsonify({'error': 'No predictions loaded'}), 500
    
    high_count = int((df['alert_level'] == 'HIGH').sum())
    medium_count = int((df['alert_level'] == 'MEDIUM').sum())
    low_count    = int((df['alert_level'] == 'LOW').sum())

    #Get best horizon results from model JSON
    optimal = {}
    if model_results and 'optimal_horizon' in model_results:
        optimal = model_results['optimal_horizon']

    return jsonify({
        'total_students': len(df),
        'high_risk':      high_count,
        'medium_risk':    medium_count,
        'low_risk':       low_count,
        'model_f1':       optimal.get('f1', 0.7227),
        'model_auc':      optimal.get('auc', 0.8831),
        'optimal_horizon': optimal.get('horizon', 'Week 17'),
        'baselines_beaten': 7
    })    


    
         








