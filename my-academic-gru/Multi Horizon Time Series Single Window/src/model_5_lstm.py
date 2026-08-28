import sys,os
sys.path.append(os.path.dirname(__file__))

import torch 
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import(f1_score,roc_auc_score,
                            precision_score,recall_score)
from data_loader import (get_data,print_metrics,get_pos_weight,find_best_threshold)

SEED = 42
MODELS_PATH = '../models/saved'
RESULTS_PATH = '../results/'
EPOCHS       = 200
PATIENCE     = 25
BATCH_SIZE   = 64
LR           = 0.0005
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'figures'), exist_ok=True)

torch.manual_seed(SEED)
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")



print("="*52)
print("MODEL 5: LSTM")
print(f"Using device: {device}")
print("="*52)

class SimpleLSTM(nn.Module):
    def __init__(self, n_features=13, hidden=HIDDEN_SIZE, num_layers=NUM_LAYERS,dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, 8_weeks, 8_features)
        lstm_out, _ = self.lstm(x)
        # Take only the last week's output
        last_week = lstm_out[:, -1, :]
        # Predict risk score
        return self.classifier(last_week)
        

#Load data

data = get_data()
X_tr_t  = torch.FloatTensor(data['X_tr']).to(device)
X_val_t = torch.FloatTensor(data['X_val']).to(device)
X_te_t  = torch.FloatTensor(data['X_te']).to(device)
y_tr_t  = torch.FloatTensor(data['y_tr']).to(device)
y_val_t = torch.FloatTensor(data['y_val']).to(device)
y_te_t  = torch.FloatTensor(data['y_te']).to(device)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE,
    shuffle=True)

#Build Model
model = SimpleLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Class weight — gives more importance to at-risk students
# because they are only 32% of data
weight     = get_pos_weight(data['y_tr'])
pos_weight = torch.tensor(
    [weight], dtype=torch.float32).to(device)

# Loss function with class weight
criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight)
 
total_params = sum(p.numel()
                   for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print(f"Class weight for at-risk: {weight:.2f}x")


# Training with validation + early stopping
print("\nTraining LSTM...")
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val F1':>6}")
print("-" * 30)

best_val_f1 = 0
best_epoch = 0
no_improve = 0
patience = 20
train_losses = []
val_f1s = []

for epoch in range(1000):
    model.train()
    epoch_loss = 0
    for X_batch,y_batch in train_loader:
        
        optimizer.zero_grad()
 
        # Forward pass — get raw logits
        logits = model(X_batch).squeeze(-1)
 
        # Calculate loss
        loss = criterion(logits, y_batch)
 
        # Backward pass — update weights
        loss.backward()
        optimizer.step()
 
        epoch_loss += loss.item()
 
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
         # Get probabilities (apply sigmoid to logits)
        val_probs = torch.sigmoid(
            model(X_val_t)
        ).squeeze().cpu().numpy()
 
        # Convert to binary predictions
        val_preds = (val_probs > 0.5).astype(int)
 
        # Calculate F1 on validation set
        val_f1 = f1_score(
            data['y_val'], val_preds,
            zero_division=0)
        val_f1s.append(val_f1)

    # Print every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:>6} | "
              f"{avg_train_loss:>10.4f} | "
              f"{val_f1:>6.4f}")

     #  Save best model 
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch + 1
        no_improve  = 0
        torch.save(model.state_dict(),
                   MODELS_PATH + 'model5_lstm_best.pt')
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            print(f"Best was epoch {best_epoch} "
                  f"(Val F1={best_val_f1:.4f})")
            break

print(f"\nTraining complete.")
print(f"Best validation F1: {best_val_f1:.4f} "
      f"at epoch {best_epoch}")        

# ── Find Best Threshold ────────────────────────────────────
# Default threshold 0.5 is not always best
# for imbalanced data. Find optimal on validation.
print("\nFinding best prediction threshold...")
 
model.load_state_dict(torch.load(
    MODELS_PATH + 'model5_lstm_best.pt',
    weights_only=True))
 
best_thresh, best_thresh_f1 = find_best_threshold(
    model, data['X_val'], data['y_val'], device)
 
print(f"Best threshold: {best_thresh:.2f} "
      f"(Val F1={best_thresh_f1:.4f})")
 

# ── Final Evaluation on Test Set ───────────────────────────
# Touch test set ONLY ONCE here at the very end
print("\nEvaluating on test set (final result)...")
 
model.eval()
with torch.no_grad():
    test_probs = torch.sigmoid(
        model(X_te_t)
    ).squeeze().cpu().numpy()
 
# Use best threshold found on validation
test_preds = (test_probs > best_thresh).astype(int)
y_true     = data['y_te']
 
f1  = f1_score(y_true, test_preds)
auc = roc_auc_score(y_true, test_probs)
pre = precision_score(y_true, test_preds,
                      zero_division=0)
rec = recall_score(y_true, test_preds,
                   zero_division=0)
 
print_metrics("Simple LSTM",
              f1, auc, pre, rec,
              can_curve=True)

# ── Save Training Curves ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Simple LSTM Training\n'
             'R26-IT-059 IT22916426',
             fontweight='bold')
 
# Training loss
axes[0].plot(train_losses,
             color='#5B9BD5',
             linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss')
axes[0].grid(True, alpha=0.3)
 
# Validation F1
axes[1].plot(val_f1s,
             color='#70AD47',
             linewidth=2)
axes[1].axvline(x=best_epoch-1,
                color='red',
                linestyle='--',
                alpha=0.7,
                label=f'Best epoch {best_epoch}')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation F1')
axes[1].set_title('Validation F1 per Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'figures/model5_lstm_training.png',
            dpi=120, bbox_inches='tight')
plt.close()
                   




