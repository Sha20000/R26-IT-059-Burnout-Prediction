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

