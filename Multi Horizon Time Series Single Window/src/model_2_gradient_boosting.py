import sys,os
sys.path.append(os.path.dirname(__file__))
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score,roc_auc_score,
                             precision_score,recall_score)
from data_loader import get_data,print_metrics

print("="*52)
print("MODEL 2: Gradient Boosting")
print("="*52)

data = get_data()
X_flat_tr = data['X_flat_tr']
X_flat_te = data["X_flat_te"]
y_tr = data['y_tr'].astype(int)
y_te = data['y_te'].astype(int)

print("\n Training Gradient Boosting")
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
model.fit(X_flat_tr, y_tr)

pred = model.predict(X_flat_te)
prob = model.predict_proba(X_flat_te)[:,1]
f1 = f1_score(y_te, pred)
auc = roc_auc_score(y_te, prob)
pre = precision_score(y_te, pred)
rec = recall_score(y_te, pred)

print_metrics("Gradient Boosting", f1, auc, pre, rec)
print("\nLIMITATION: No temporal learning. No curve.")