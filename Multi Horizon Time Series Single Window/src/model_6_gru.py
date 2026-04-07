"""
model6_gru.py
=============
R26-IT-059 | IT22916426 | Mahavitha S.M.

MODEL 6: GRU (Gated Recurrent Unit)
PURPOSE: Simpler recurrent model than LSTM.
         Fewer parameters, similar performance.
         Still single output — cannot produce curve.

RUN:
    python model6_gru.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, roc_auc_score,
                              precision_score, recall_score)
from data_loader import (get_data, print_metrics,
                          get_pos_weight,
                          find_best_threshold)

# ── Configuration ──────────────────────────────────────────
SEED        = 42
MODELS_PATH = '../models/saved/'
EPOCHS      = 200
PATIENCE    = 25
BATCH_SIZE  = 64
LR          = 0.0005
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3

os.makedirs(MODELS_PATH, exist_ok=True)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("=" * 52)
print("MODEL 6: GRU (Baseline)")
print(f"Device: {device}")
print("=" * 52)


# ── Model Architecture ─────────────────────────────────────
class SimpleGRU(nn.Module):
    """
    GRU — simpler than LSTM.
    Uses reset and update gates (not cell state).
    Often similar accuracy with fewer parameters.
    Still single output — cannot produce curve.
    """

    def __init__(self,
                 n_features=13,
                 hidden=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.classifier(gru_out[:, -1, :])


#  Load Data 
data    = get_data()
X_tr_t  = torch.FloatTensor(data['X_tr']).to(device)
X_val_t = torch.FloatTensor(data['X_val']).to(device)
X_te_t  = torch.FloatTensor(data['X_te']).to(device)
y_tr_t  = torch.FloatTensor(data['y_tr']).to(device)
y_val_t = torch.FloatTensor(data['y_val']).to(device)

loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE, shuffle=True)

#  Build Model 
model     = SimpleGRU().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

weight     = get_pos_weight(data['y_tr'])
pos_weight = torch.tensor(
    [weight], dtype=torch.float32).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"\nParameters: "
      f"{sum(p.numel() for p in model.parameters()):,}")
print(f"Class weight: {weight:.2f}x")

#  Training 
print(f"\nTraining GRU...")
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val F1':>6}")
print("-" * 30)

best_val_f1 = 0
best_epoch  = 0
no_improve  = 0

for epoch in range(EPOCHS):
    model.train()
    ep_loss = 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb).squeeze(-1), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()

    model.eval()
    with torch.no_grad():
        vp  = torch.sigmoid(
            model(X_val_t)).squeeze().cpu().numpy()
        vf1 = f1_score(data['y_val'],
                       (vp > 0.5).astype(int),
                       zero_division=0)

    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:>6} | "
              f"{ep_loss/len(loader):>10.4f} | "
              f"{vf1:>6.4f}")

    if vf1 > best_val_f1:
        best_val_f1 = vf1
        best_epoch  = epoch + 1
        no_improve  = 0
        torch.save(model.state_dict(),
                   MODELS_PATH + 'model6_gru_best.pt')
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch+1} "
                  f"(best={best_epoch})")
            break

# ── Evaluate ───────────────────────────────────────────────
model.load_state_dict(torch.load(
    MODELS_PATH + 'model6_gru_best.pt',
    weights_only=True))

best_thresh, _ = find_best_threshold(
    model, data['X_val'], data['y_val'], device)

model.eval()
with torch.no_grad():
    pred = torch.sigmoid(
        model(X_te_t)).squeeze().cpu().numpy()

yb  = (pred > best_thresh).astype(int)
f1  = f1_score(data['y_te'], yb)
auc = roc_auc_score(data['y_te'], pred)
pre = precision_score(data['y_te'], yb, zero_division=0)
rec = recall_score(data['y_te'], yb, zero_division=0)

print_metrics("GRU", f1, auc, pre, rec, can_curve=False)
print(f"Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")