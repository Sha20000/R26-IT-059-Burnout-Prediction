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

#Step 8 : FNN

class MetaFNN(nn.Module):

    """
    Meta-Integration FNN.
    Input:  10 features (4 GRU + 6 VAE)
    Output: final burnout risk (0.0-1.0)
    """
    def __init__(self,n_features = 10, hidden = HIDDEN_SIZE, dropout = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),

            nn.Linear(hidden // 4, 1)
        )

    def forward(self,x):
        return self.net(x)



model = MetaFNN(n_features=len(FEATURE_COLS))
optimizer = torch.optim.Adam(
    model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10
)

pos_weight = torch.tensor(
    [(1 - y_train.mean()) / y_train.mean()],
    dtype=torch.float32
)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"\nMeta FNN parameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")
print(f"Class weight: {pos_weight.item():.2f}x")


#Step 9 : Train


print(f"\nTraining...")
print(f"{'Epoch':>6} | {'Loss':>8} | "
      f"{'Val F1':>8} | {'Best':>8}")
print("-" * 42)

best_f1 = 0
best_epoch = 0
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b).squeeze(-1), y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(
            model(X_val_t).squeeze(-1)).numpy()
        vf = f1_score(y_val, (vp > 0.5).astype(int),
                      zero_division=0)

    scheduler.step(vf)

    if (epoch + 1) % 20 == 0:
        print(f"{epoch+1:>6} | {avg_loss:>8.4f} | "
              f"{vf:>8.4f} | {best_f1:>8.4f}")

    if vf > best_f1:
        best_f1    = vf
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(),
                   os.path.join(OUTPUT_DIR,
                                'meta_fnn_best.pt'))
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stop epoch {epoch+1} "
                  f"(best: {best_epoch})")
            break

print(f"\nBest val F1: {best_f1:.4f} at epoch {best_epoch}")

#Step 10: Test

print("\n" + "=" * 60)
print("TEST SET EVALUATION")
print("=" * 60)

model.load_state_dict(torch.load(
    os.path.join(OUTPUT_DIR, 'meta_fnn_best.pt'),
    weights_only=True))
model.eval()

with torch.no_grad():
    test_probs = torch.sigmoid(
        model(X_te_t).squeeze(-1)).numpy()

# Best threshold from validation
best_thresh = 0.5
best_tf1    = 0
for t in np.arange(0.20, 0.71, 0.05):
    with torch.no_grad():
        vp = torch.sigmoid(
            model(X_val_t).squeeze(-1)).numpy()
    f = f1_score(y_val, (vp > t).astype(int),
                 zero_division=0)
    if f > best_tf1:
        best_tf1    = f
        best_thresh = t

test_preds     = (test_probs > best_thresh).astype(int)
f1   = f1_score(y_test,  test_preds)
auc  = roc_auc_score(y_test, test_probs)
prec = precision_score(y_test, test_preds, zero_division=0)
rec  = recall_score(y_test,  test_preds,  zero_division=0)
acc  = accuracy_score(y_test, test_preds)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()

print(f"\nThreshold:  {best_thresh:.2f}")
print(f"Accuracy:   {acc*100:.2f}%")
print(f"F1 Score:   {f1:.4f}")
print(f"AUC-ROC:    {auc:.4f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")
print()
print(f"  TRUE  POSITIVE: {tp:>4}  at-risk caught")
print(f"  TRUE  NEGATIVE: {tn:>4}  safe cleared")
print(f"  FALSE POSITIVE: {fp:>4}  false alarms")
print(f"  FALSE NEGATIVE: {fn:>4}  missed at-risk")





