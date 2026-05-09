import os;
import json;
import numpy as np;
import pandas as pd;
import torch;
import torch.nn as nn;
from torch.utils.data import DataLoader,TensorDataset;
from sklearn.metrics import(
    f1_score, roc_auc_score, 
    precision_score, recall_score,
    accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split;
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


#Configurations
SEED        = 42
EPOCHS      = 300
PATIENCE    = 40
BATCH_SIZE  = 64
LR          = 0.001
HIDDEN_SIZE = 64
DROPOUT     = 0.3

torch.manual_seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED) 
np.random.seed(SEED)    

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "meta_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#CSV paths
ACADEMIC_CSV  = os.path.join(
    DATA_DIR, 'model8_predictions.csv')
BEHAVIOR_CSV  = os.path.join(
    DATA_DIR, 'behavior_prediction.csv')

print("=" * 60)
print("R26-IT-059 | Meta-Integration Layer")
print("GRU Academic Risk + VAE Behavioural Risk")
print("=" * 60)

#Step 1: Load CSVS

print("\n Loading CSVs...")
academic_df = pd.read_csv(ACADEMIC_CSV)
behavior_df = pd.read_csv(BEHAVIOR_CSV)

print(f"  Academic  (IT22916426): {len(academic_df)} rows")
print(f"  Behaviour (IT22215710): {len(behavior_df)} rows")
print(f"\n  Behaviour columns: {list(behavior_df.columns)}")

# Step 2: Aggregate behaviour per student


print("\nAggregating behaviour per student...")

behavior_agg = behavior_df.groupby('student_id').agg(
    # Average behavioural risk across all weeks
    behavior_risk_mean  = ('behavioral_risk_score', 'mean'),
    # Peak behavioural risk (worst week)
    behavior_risk_max   = ('behavioral_risk_score', 'max'),
    # How many weeks flagged as high anomaly
    anomaly_weeks       = ('high_anomaly_flag', 'sum'),
    # Average compliance with curriculum
    compliance_mean     = ('curriculum_compliance', 'mean'),
    # Minimum compliance (worst week compliance)
    compliance_min      = ('curriculum_compliance', 'min'),
    # Average raw anomaly score
    anomaly_score_mean  = ('anomaly_score', 'mean'),
    # Peak raw anomaly score
    anomaly_score_max   = ('anomaly_score', 'max'),
    # Total weeks observed
    total_weeks         = ('week', 'count'),
    # Confidence average
    confidence_mean     = ('confidence_score', 'mean'),
).reset_index()

print(f"  Aggregated: {len(behavior_agg)} unique students")

#Step 3 Merge on student id
print("\nMerging academic + behaviour...")

# Check types
print(f"  Academic student_id type:  {academic_df['student_id'].dtype}")
print(f"  Behaviour student_id type: {behavior_agg['student_id'].dtype}")

# Convert both to string for safe merge
academic_df['student_id']   = academic_df['student_id'].astype(str)
behavior_agg['student_id']  = behavior_agg['student_id'].astype(str)


merged = academic_df[[
    'student_id',
    'academic_risk',
    'week4_risk',
    'week8_risk',
    'week12_risk',
    'week17_risk',
    'actual_label'
]].merge(behavior_agg, on='student_id', how='left')

print(f"  Merged shape: {merged.shape}")
print(f"\n  Missing after merge:")
print(merged.isnull().sum())

#step 4 : handld missing values

behavior_cols = [
    'behavior_risk_mean', 'behavior_risk_max',
    'anomaly_weeks', 'compliance_mean',
    'compliance_min', 'anomaly_score_mean',
    'anomaly_score_max', 'confidence_mean'
]

for col in behavior_cols:
    median_val     = merged[col].median()
    merged[col]    = merged[col].fillna(median_val)


merged['total_weeks'] = merged['total_weeks'].fillna(0)

print("\n  After imputation — missing values:")
print(merged[behavior_cols].isnull().sum())

#Step 5: feature matrix

FEATURE_COLS = [
    # ── GRU signals (IT22916426) ──────────────────────────
    'academic_risk',      # overall Week 17 risk
    'week4_risk',         # early signal
    'week8_risk',         # mid signal
    'week12_risk',        # late signal

    # ── VAE signals (IT22215710) ──────────────────────────
    'behavior_risk_mean', # avg behaviour risk all weeks
    'behavior_risk_max',  # worst week behaviour risk
    'anomaly_score_mean', # avg raw anomaly score
    'compliance_mean',    # avg curriculum compliance
    'compliance_min',     # worst compliance week
    'anomaly_weeks',      # count of high anomaly weeks
]

X = merged[FEATURE_COLS].values.astype(np.float32)
y = merged['actual_label'].values.astype(np.float32)

print(f"\nFeature matrix: {X.shape}")
print(f"  {len(FEATURE_COLS)} features: 4 GRU + 6 VAE")
print(f"\nLabel distribution:")
print(f"  At-risk (1): {int(y.sum())}  ({y.mean()*100:.1f}%)")
print(f"  Safe    (0): {int((1-y).sum())}  ({(1-y).mean()*100:.1f}%)")


#Step 6 Split

X_tmp,  X_test, y_tmp,  y_test = train_test_split(
    X, y, test_size=0.15,
    random_state=SEED, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.15,
    random_state=SEED, stratify=y_tmp)

print(f"\nSplit:")
print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}")



#Step 7 : Scale

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

#Save scaler for API use
with open(os.path.join(
        OUTPUT_DIR, 'meta_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

X_tr_t  = torch.FloatTensor(X_train_s)
X_val_t = torch.FloatTensor(X_val_s)
X_te_t  = torch.FloatTensor(X_test_s)
y_tr_t  = torch.FloatTensor(y_train)
y_val_t = torch.FloatTensor(y_val)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE, shuffle=True)




