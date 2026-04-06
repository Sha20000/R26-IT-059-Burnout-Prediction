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
from data_loader import get_data,print_metrics

SEED = 42
MODELS_PATH = '../models/saved'
RESULTS_PATH = '../results/'
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH,'figures',exist_ok=True)

torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   



print("="*52)
print("MODEL 5: LSTM")
print(f"Using device: {device}")
print("="*52)

class SimpleLSTM(nn.Module):

    def __init__(self, n_features = 8, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,1),
            nn.Sigmoid()
        )

        def forward(self,x):
            out, _ = self.lstm(x)
            return self.fc(out[:,-1,:])
        

#Load data

data = get_data()
X_tr_t  = torch.FloatTensor(data['X_tr']).to(device)
X_val_t = torch.FloatTensor(data['X_val']).to(device)
X_te_t  = torch.FloatTensor(data['X_te']).to(device)
y_tr_t  = torch.FloatTensor(data['y_tr']).to(device)
y_val_t = torch.FloatTensor(data['y_val']).to(device)
y_te_t  = torch.FloatTensor(data['y_te']).to(device)

loader = DataLoader(
    TensorDataset(X_tr_t,y_tr_t), 
                    batch_size=64, shuffle=True)

#Build Model
model = SimpleLSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training with validation + early stopping
print("\nTraining LSTM...")
print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Val F1':>6}")
print("-" * 30)

best_val_f1 = 0
best_epoch = 0
no_improve = 0
patience = 10
train_losses = []
val_f1s = []

for epoch in range(80):
    model.train()
    ep_loss = 0
    for Xb,yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb).squeeze(), yb)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item()

    tl = ep_loss / len(loader)
    train_losses.append(tl)

    model.eval()
    with torch.no_grad():
        vp = model(X_val_t).squeeze().cpu().numpy()
        vf1 = f1_score(data['y_val'], vp > 0.5.astype(int))
        val_f1s.append(vf1)

    if (epoch+1)%10 == 0:
        print(f"{epoch+1:>6} | {tl:>10.4f} | {vf1:>6.4f}")

    if vf1 > best_val_f1:
        best_val_f1 = vf1
        best_epoch = epoch+1
        no_improve = 0
        torch.save(model.state_dict(),
                   MODELS_PATH+ "lstm_best.pt")

    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stop at epoch {epoch+1}"
                  f" (best={best_epoch})")
            break

#Load best and evaluate test

model.load_state_dict( torch.load(
    MODELS_PATH+ "lstm_best.pt",
    weights_only=True)
)   

model.eval()
with torch.no_grad():
    pred = model(X_te_t).squeeze().cpu().numpy()
    y_np = data['y_te']
    yb =  (pred > 0.5).astype(int)


f1 = f1_score(y_np, yb)
auc = roc_auc_score(y_np, pred)
pre = precision_score(y_np, yb)
rec = recall_score(y_np, yb)

print_metrics("LSTM", f1, auc, pre, rec,can_curve=True)
print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch}")

#Save Training curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss',
         color='#5B9BD5', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Simple LSTM — Training Loss\nR26-IT-059 IT22916426',
          fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_PATH + 'figures/lstm_training.png',
            dpi=120, bbox_inches='tight')
print("Training curve saved.")
print("\nLIMITATION: Single horizon only.")
print("Cannot answer: WHEN to intervene?")
                   




