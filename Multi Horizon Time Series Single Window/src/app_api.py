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
    model_results = {}
    print("WARNING: model results JSON not found")

print(f"API ready at http://localhost:5001")
print("=" * 50)   



