import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn 
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from data_loader import (get_data, print_metrics, get_pos_weight, find_best_threshold)

#Configuration
SEED        = 42
MODELS_PATH = '../models/saved/'
RESULTS_PATH = '../results/'
EPOCHS     = 300
PATIENCE  = 30
BATCH_SIZE = 64
LR        = 0.0003
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT    = 0.3

# 4 prediction horizons - week indices (0 based)
# Week 4 = index 3
# Week 8 = index 7
# Week 12 = index 11
# Week 17 = index 16

HORIZON_WEEKS = [4,8,12,17]
HORIZON_INDICES = [3,7,11,16]
HORIZON_LABELS = ['Week 4', 'Week 8', 
                  'Week 12', 'Week 17']

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'figures'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_PATH, 'metrics'), exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

#Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("=" * 60)
print("MODEL 8: Unified Multi-Horizon GRU")
print("R26-IT-059 | IT22916426 | Mahavitha S.M.")
print(f"Device: {device}")
print()
print("NOVELTY:")
print("  4 simultaneous prediction heads")
print("  Attention mechanism")
print("  XAI feature attribution")
print("  Accuracy-vs-lead-time curve")
print("=" * 60)

#Model Architecture

class AttentionLayer(nn.Module):
     
    def __init__(self, hidden_size):
        super().__init__()
          # Learns to score each week's importance
        self.attention_weights = nn.Linear(
             hidden_size,1
        ) 

    def forward(self,gru_output):

        # gru_output shape: (batch, weeks, hidden)

        #Calculate importance scores for each week

        scores = self.attention_weights(
            gru_output
        )     # (batch, weeks, 1)

        #Convert to probabilities (sum to 1)

        weights = torch.softmax(scores,
                                 dim=1) # (batch, weeks, 1)
        
        # Weighted sum of all weekly states

        attended = (weights*gru_output).sum(
            dim=1
        ) # (batch, hidden)

        return attended, weights.squeeze(-1) # Return weights for XAI
    
class PredictionHead(nn.Module):

    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.head(x)

class UnifiedMultiHorizonGRU(nn.Module):


    def __init__(self,
                      n_features,
                       hidden =HIDDEN_SIZE,
                        num_layers=NUM_LAYERS,
                         dropout=DROPOUT,
                          horizon_indices=HORIZON_INDICES ):
        super().__init__()   

        self.horizon_indices = horizon_indices
        self.n_horizons = len(horizon_indices)

        #GRU Encoder 
        # Reads weekly sequence and produces
        # hidden state at each week

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        ) 

        # Attention layer 
        #Learns which weeks matter most 
        self.attention = AttentionLayer(hidden)

        #Prediction heads 
        # One head per prediction horizon

        self.heads = nn.ModuleList([
            PredictionHead(hidden,dropout)
            for _ in range(self.n_horizons)
        ]) 

        #Dropout

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): 
        """
        Forward pass.

        INPUT:
            x shape: (batch, 17_weeks, 13_features)

        OUTPUT:
            predictions: list of 4 tensors
                         each shape (batch, 1)
            attention_weights: (batch, 17_weeks)
        """
        
        # Step 1: GRU reads entire sequence
        # gru_out shape: (batch, 17, hidden)

        gru_out, _ = self.gru(x)

        #Step 2 : Attention over all weeks
        #attended shape: (batch, hidden)

        attended, attn_weights = self.attention(gru_out)

        #Step 3: Each head predicts at its horizon
        predictions = []

        for i,(head,idx) in enumerate(
            zip(self.heads, self.horizon_indices)):
            # Get GRU state at this specific week
            # Combined with attention context

            week_state = gru_out[:, idx, :]
            week_state = self.dropout(week_state) # Regularize

            #Predict risk at this horizon
            pred = head(week_state)
            predictions.append(pred)

        return predictions, attn_weights


#LOAD DATA

print("\nLoading data...")
data   = get_data()

#Get actual feature count from data 
N_FEATURES = data['X_tr'].shape[2]
N_WEEKS = data['X_tr'].shape[1]

print(f"Features:{N_FEATURES}")
print(f"Weeks: {N_WEEKS}")

#Convert to tensors 
X_tr_t  = torch.FloatTensor(data['X_tr']).to(device)
X_val_t = torch.FloatTensor(data['X_val']).to(device)
X_te_t  = torch.FloatTensor(data['X_te']).to(device)
y_tr_t  = torch.FloatTensor(data['y_tr']).to(device)
y_val_t = torch.FloatTensor(data['y_val']).to(device)
y_te_t  = torch.FloatTensor(data['y_te']).to(device)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t),
    batch_size=BATCH_SIZE, 
    shuffle=True
)

#Build Model

model = UnifiedMultiHorizonGRU(
    n_features=N_FEATURES.to(device),
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=LR,weight_decay=1e-5)


)

#Learning rate schedular
#Reduce LR when validation stops improving
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,mode='max',
    factor=0.5,patience=10
)

#Class weight for imbalanced data
weight = get_pos_weight(data['y_tr'])
pos_weight = torch.tensor(
    [weight], dtype=torch.float32
).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print(f"Class weight:{weight:.2f}x")
print(f"Horizons: {HORIZON_LABELS}")

#Training Loop
print(f"\nTraining Unified Multi-Horizon GRU...")
print(f"Up to {EPOCHS} epochs with early stopping (patience={PATIENCE})")
print()
print(f"{'Epoch':>6} | {'Loss':>8}| "
      f"{'H1-W4':>7} | {'H2-W8':>7} | "
      f"{'H3-W12':>7} | {'H4-W17':>7} | "
       f"{'Best':>7}")
print("-" * 60)

best_val_f1 = 0
best_epoch = 0
no_improve = 0
train_losses = []
val_f1s_all = [[] for _ in range(4)]
val_f1s_avg = []

for epoch in range(EPOCHS):

    #Training
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        #Forward pass - get 4 predictions
        predictions,_ = model(X_batch)

        #Calculate loss for each head
        #All heads predict same label

        total_loss = 0
        for pred in predictions:
            loss = criterion(
                pred.squeeze(-1),y_batch
            )
            total_loss += loss


        #Average loss across 4 heads

        total_loss = total_loss / len(predictions)

        total_loss.backward()  

        #Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),max_norm=1.0
        )  

        optimizer.step()
        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    #Validation

    model.eval()
    horizon_f1s = []

    with torch.no_grad():

        val_preds,_ = model(X_val_t)

        for i,pred in enumerate(val_preds):

            probs = torch.sigmoid(pred).squeeze().cpu().numpy()

            preds = (probs > 0.5).astype(int)

            f1 = f1_score(data['y_val'], preds, zero_division=0)

            horizon_f1s.append(f1)

            val_f1s_all[i].append(f1)

    avg_f1 = np.mean(horizon_f1s)
    val_f1s_avg.append(avg_f1)

    #Update learning rate schedular
    schedular.step(avg_f1)

    #Print progress every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f"{epoch+1:>6} |"
              f"{avg_loss:>8.4f} | "
              f"{horizon_f1s[0]:>7.4f} | "
              f"{horizon_f1s[1]:>7.4f} | "
              f"{horizon_f1s[2]:>7.4f} | "
              f"{horizon_f1s[3]:>7.4f} | "
              f"{avg_f1:>7.4f}")

    #Save best model
    if avg_f1 > best_val_f1:
        best_val_f1 = avg_f1
        best_epoch = epoch + 1
        no_improve = 0
        torch.save(model.state_dict(),
                   MODELS_PATH + 'model8_unified_gru_best.pt')

    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(best epoch was {best_epoch})")
            break


print("\nTraining complete.")
print(f"Best avg val F1: {best_val_f1:.4f} at epoch {best_epoch}")

#FINAL EVALUATION - TEST SET
print("\n"+"="*60)
print("Evaluating on test set...")
print("="*60)

model.load_state_dict(torch.load(MODELS_PATH + 'model8_unified_gru_best.pt', 
                                 weights_only=True))
model.eval()

results = []
all_attn_weights = []

with torch.no_grad():
    test_preds, attn_weights = model(X_te_t)
    all_attn_weights = attn_weights.cpu().numpy()

print(f"\n{'Horizon':<12}| {'F1':>6} |"
      f"{'AUC':>6} | {'Prec':>6} | "
      f"{'Rec':>7}| {'Threshold':>9}")
print("-" * 60)

best_horizon_idx = 0
best_horizon_f1 = 0

for i, (pred,label) in enumerate(
    zip(test_preds,HORIZON_LABELS)):

    probs = torch.sigmoid(pred).squeeze().cpu().numpy()
    y_true = data['y_te']

    #Find best threshold on validation for this horizon

    val_pred_i = torch.sigmoid(
        model(X_val_t)[0][i]).squeeze().cpu().numpy()
    
    best_thresh = 0.5
    best_f1_v = 0
    for t in np.arrange(0.2,0.71,0.05):
        p = (val_pred_i > t).astype(int)
        f = f1_score(data['y_val'], p,
                      zero_division=0)
        
        if f > best_f1_v:
            best_f1_v = f
            best_thresh = t

    preds = (probs > best_thresh).astype(int)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    pre = precision_score(y_true, preds,
                          zero_division=0)
    rec = recall_score(y_true, preds,zero_division=0)

    print(f"{label:<12}| {f1:>6.4f} | "
          f"{auc:>6.4f} | {pre:>6.4f} | "
          f"{rec:>6.4f} | {best_thresh:>9.2f}")    

    results.append({
        'horizon': label,
        'f1': f1,
        'auc': auc,
        'precision': pre,
        'recall': rec,
        'threshold': best_thresh
    })    

    if f1 > best_horizon_f1:
        best_horizon_f1 = f1
        best_horizon_idx = i


print()
print(f"OPTIMAL HORIZON:"
      f"{results[best_horizon_idx]['horizon']} ")
print(f" OPTIMAL F1: {results[best_horizon_idx]['f1']:.4f} | ")
print(f" AUC: {results[best_horizon_idx]['auc']:.4f} | ")







        





        
           
          
                   
        





































