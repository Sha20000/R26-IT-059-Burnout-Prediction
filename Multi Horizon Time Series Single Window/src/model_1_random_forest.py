import sys,os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib 
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score,)
from data_loader import get_data, print_metrics

RESULTS_PATH = '../results/'
os.makedirs(RESULTS_PATH, exist_ok=True)

print("="*52)
print("MODEL 1: Random Forest")
print("="*52)

# Load data
data = get_data()
X_flat_tr = data['X_flat_tr']
X_flat_te = data['X_flat_te']
y_tr = data['y_tr'].astype(int)
y_te = data['y_te'].astype(int)

#Train model
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_flat_tr, y_tr)
print("Training complete.")

#Evaluate on test set
pred = model.predict(X_flat_te)
prob = model.predict_proba(X_flat_te)[:,1]

f1 = f1_score(y_te, pred)
auc = roc_auc_score(y_te, prob)
pre = precision_score(y_te, pred)
rec = recall_score(y_te, pred)

print_metrics("Random Forest", f1, auc, pre, rec, can_curve=False)

# Feature importance chart
feat_names = ['F1_logins', 'F2_clicks', 'F3_score',
               'F4_num_ass','F5_ontime','F6_inactive',
               'F7_trend','F8_cumulative']

#Average importance across all week columns
avg_imp = np.zeros(8)
for i in range(8):
    avg_imp[i] = model.feature_importances_[i::8].mean()
avg_imp = avg_imp/avg_imp.sum()






