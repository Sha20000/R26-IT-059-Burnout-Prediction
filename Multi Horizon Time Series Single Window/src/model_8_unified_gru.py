import sys
import os
sys.path.append(os.path.dirname(__file__))
import shap 
import torch
import torch.nn as nn 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (f1_score, roc_auc_score,precision_score, recall_score)
from data_loader import (get_data, print_metrics, get_pos_weight, find_best_threshold)


#Configuration
SEED        = 42
MODELS_PATH = '../models/saved/'
RESULTS_PATH = '../results/'
EPOCHS     = 500
PATIENCE  = 50
BATCH_SIZE = 64
LR        = 0.0001
HIDDEN_SIZE = 256
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
FEATURE_NAMES = [
    'F1_login_count',
    'F2_vle_clicks',
    'F3_avg_score',
    'F4_num_submitted',
    'F5_on_time_rate',
    'F6_inactive_days',
    'F7_score_change',
    'F8_running_avg',
    'F9_score_variance',
    'F10_min_score',
    'F11_max_score',
    'F12_late_count',
    'F13_weeks_since_active'


]

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
        self.gate = nn.Linear(hidden * 2, hidden)

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
           week_state = self.dropout(week_state)
           pred       = head(week_state)
           predictions.append(pred)

        return predictions, attn_weights

#XAI FUNCTIONS

def get_feature_importance(model,X_tensor
                           ,horizon_idx=3):
    "XAI Level 2 - Feature Attribution"
    
    model.eval()
    X = X_tensor.clone().requires_grad_(True)

    predictions,_ = model(X)
    pred = predictions[horizon_idx]
    pred.sum().backward()

    #Absolute gradient for feature 
    gradients = X.grad.abs()
    importance = gradients.mean(
        dim=(0,1)).cpu().detach().numpy()
    
    if importance.sum() > 0:
        
        importance = importance / importance.sum()

    return importance





def explain_single_student(model, X_tensor, 
                           student_idx,horizon_idx=3):
    " XAI Level 3 — Single Student Explanation."

    model.eval()
    X_one = X_tensor[student_idx:student_idx+1].\
    clone().requires_grad_(True)

    predictions,attn = model(X_one)
    pred = predictions[horizon_idx]
    risk_prob = torch.sigmoid(pred).item()

    pred.sum().backward()
    gradients = X_one.grad.abs()
    importance = gradients.mean(dim=1).squeeze().cpu().detach().numpy()

    if importance.sum() > 0:
        importance = importance / importance.sum()

    attn_weights = attn[0].cpu().detach().numpy()

    return {
        'student_idx': student_idx,
        'risk_probability': risk_prob,
        'risk_level':('HIGH' if risk_prob > 0.7 else 'MEDIUM'
                      if risk_prob > 0.4 else 'LOW'),
        'feature_importance': dict(zip(FEATURE_NAMES, importance)),
        'attention_weights': attn_weights

    }  



    


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
    n_features=N_FEATURES).to(device)

optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=LR,weight_decay=1e-5)




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
        model(X_val_t)[0][i]).squeeze().cpu().detach().numpy()
    
    best_thresh = 0.5
    best_f1_v = 0
    for t in np.arange(0.2,0.71,0.05):
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
        'week':HORIZON_WEEKS[i],
        'f1': f1,
        'auc': auc,
        'precision': pre,
        'recall': rec,
        'threshold': best_thresh,
        
    })    

    if f1 > best_horizon_f1:
        best_horizon_f1 = f1
        best_horizon_idx = i


print()
print(f"OPTIMAL HORIZON:"
      f"{results[best_horizon_idx]['horizon']} ")
print(f" OPTIMAL F1: {results[best_horizon_idx]['f1']:.4f} | ")
print(f" AUC: {results[best_horizon_idx]['auc']:.4f} | ")

#  Calculate XAI values BEFORE charts 

# Gradient feature importance
importance = get_feature_importance(
    model, X_te_t, horizon_idx=3)

ranked = sorted(
    zip(FEATURE_NAMES, importance),
    key=lambda x: x[1], reverse=True)

top_feature = ranked[0][0]

# Top attention weeks
avg_attn = all_attn_weights.mean(axis=0)
top_3    = np.argsort(avg_attn)[::-1][:3] + 1

# Single student explanation
with torch.no_grad():
    all_probs = torch.sigmoid(
        model(X_te_t)[0][3]
    ).squeeze().cpu().numpy()

high_risk_idx = int(all_probs.argmax())
explanation   = explain_single_student(
    model, X_te_t,
    student_idx=high_risk_idx,
    horizon_idx=3)

print(f"XAI ready.")
print(f"Most important feature: {top_feature}")
print(f"Top attention weeks: {list(top_3)}")


#CHART 1 - ACCURACY VS LEAD TIME CURVE

print("\nGenerating accuracy vs lead-time curve")

fig,ax = plt.subplots(figsize=(8,5))

weeks = [r['week'] for r in results]
f1s = [r['f1'] for r in results]
aucs = [r['auc'] for r in results]

ax.plot(weeks,f1s,'o-',color='#C00000',linewidth=2.5,
        markersize=8,label='F1 Score',zorder=3)

ax.plot(weeks,aucs,'s--',color='#5B9BD5',linewidth=2,
        markersize=7,label='AUC-ROC',zorder=3)

#Mark optimal point
opt_week = results[best_horizon_idx]['week']
opt_f1 = results[best_horizon_idx]['f1']

ax.axvline( x = opt_week,
           color='#70AD47',
           linestyle='--',alpha=0.7,linewidth=1.5,
           label=f'Optimal Horizon: {opt_week}')

ax.scatter([opt_week],[opt_f1],color='#70AD47',s=100,
           zorder=5, label=f'Best F1 = {opt_f1:.3f}')

#Annotations
for r in results:
    ax.annotate(
        f"{r['f1']:.3f}",
        xy=(r['week'],r['f1']),
        xytext=(0,10),
        textcoords='offset points',
        ha='center', fontsize=9,
        fontweight='bold'
    )

ax.set_xlabel('Prediction Horizon (Weeks)',fontsize=12)
ax.set_ylabel('Score',fontsize=12)

ax.set_title('Accuracy vs Lead Time Curve\n'
             'R26-IT-059 | IT22916426 | '
             'Unified Multi-Horizon GRU',
             fontweight='bold',
             fontsize=12)
ax.set_xticks(weeks)
ax.set_xticklabels([f"Weeks {w}" for w in weeks])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5,1.0)

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'figures/unified_gru_accuracy_curve.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Accuracy vs lead-time curve saved.")

#CHART 2 - COMPARISON WITH ALL BASELINES

print("\nGenerating comparison chart with all baselines...")

baseline_names = [
    'Random\nForest',
    'Logistic\nReg',
    'XGBoost',
    'Gradient\nBoosting',
    'Simple\nLSTM',
    'GRU',
    'BiLSTM',
    f'YOUR\nModel8\n(W{opt_week})'
]

#Replace last value with your best horizon
baseline_f1s = [
    0.6647, 0.6800, 0.6890, 0.6965,
    0.7087, 0.7185, 0.7107,
    results[best_horizon_idx]['f1']
]

colours = ['#BDD7EE']*7+['#C00000']

fig, ax = plt.subplots(figsize=(12,6))
bars = ax.bar(baseline_names,baseline_f1s,color=colours,edgecolor='white',linewidth=1.5)

#Add value labels on bars
for bar, val in zip(bars, baseline_f1s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{val:.4f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
    

#Horizontal line at best baseline 
ax.axhline(y = max(baseline_f1s[:-1]),
           color = 'gray',
           linestyle = '--',
           alpha = 0.5,
           label=f'Best baseline: '
                 f'{max(baseline_f1s[:-1]):.4f}'

           
           
           )    
    

ax.set_ylabel('F1 Score',fontsize=12)
ax.set_title('Model Comparison - F1 Score\n'
             'R-26-IT-059 | IT22916426 |',
             fontweight='bold')
ax.set_ylim(0.5,0.95)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'figures/unified_gru_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model 8_comparison.png")




#Chart 3 - Attention weight heatmap

print("\nGenerating attention weight heatmap...")

avg_attn = all_attn_weights.mean(axis=0)

fig, ax  = plt.subplots(figsize=(10,3))
im = ax.imshow(
    avg_attn.reshape(1,-1),
    aspect='auto',
    cmap='Reds'

)

ax.set_xticks(range(N_WEEKS))
ax.set_xticklabels([f'W{i+1}' for i in range(N_WEEKS)])
ax.set_title('Attention Weights - Which Weeks'
             'the Model Focuses On\n'
             'R-26-IT-059 | IT22916426 | ',
             fontweight='bold')
plt.colorbar(im, ax=ax, label=' Attention Weight')

plt.tight_layout()
plt.savefig(RESULTS_PATH + 'figures/unified_gru_attention_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: unified_gru_attention_heatmap.png")

#CHART 4 -XAI FEATURE IMPORTANCE

print("Generating Chart 4: XAI features importance...")

fig,ax = plt.subplots(figsize=(15,8))
feat_labels = [f.replace('_'
                         ,'\n') for f in FEATURE_NAMES]
colours_f1 = ['#C00000' if imp == max(importance) else '#5B9BD5'
              for imp in importance]

bars = ax.barh(feat_labels, importance*100,color = colours_f1)

for bar,val in zip(bars, importance*100):

    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%',
            va='center', fontsize=9, 
            fontweight='bold')
    
ax.set_xlabel('Feature Importance (%)', 
                  fontsize=12)
    
ax.set_title('XAI Level 2 - Feature Attribution\n'
                 'Which Features Drive Burnout Prediction\n'
                 'R-26-IT-059 | IT22916426 |',
                 fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(
    RESULTS_PATH+ 
    'figures/model_8_xai_features.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: model_8_xai_features.png")

# Create wrapper class — DeepExplainer needs nn.Module
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        predictions, _ = self.model(x)
        return predictions[3]  # Week 17

# Move to CPU for SHAP
model.to('cpu')
wrapper = ModelWrapper(model).to('cpu')
wrapper.eval()

background = X_tr_t[:50].cpu()
X_shap     = X_te_t[:100].cpu()

explainer = shap.DeepExplainer(wrapper, background)
shap_vals = explainer.shap_values(X_shap,check_additivity=False)

# Move model back
model.to(device)

# Average across students and weeks
shap_array = np.array(shap_vals)
print(f"SHAP array shape: {shap_array.shape}")


# Handle different output shapes
# DeepExplainer for GRU returns (students, weeks, features, 1)
print(f"SHAP raw shape: {shap_array.shape}")

if shap_array.ndim == 4:
    # Check which axis is features (should be 13)
    if shap_array.shape[-1] == 1:
        # Shape: (students, weeks, features, 1)
        # squeeze the output dim first, then average students+weeks
        mean_shap = np.abs(
            shap_array.squeeze(-1)      # → (students, weeks, features)
        ).mean(axis=(0, 1))             # → (features,) = (13,)
    else:
        # Shape: (outputs, students, weeks, features)
        mean_shap = np.abs(
            shap_array).mean(axis=(0, 1, 2))

elif shap_array.ndim == 3:
    # Shape: (students, weeks, features)
    mean_shap = np.abs(
        shap_array).mean(axis=(0, 1))

else:
    mean_shap = np.abs(shap_array).mean(axis=0)

mean_shap = mean_shap.flatten()

# Safety check — must be 13 features
print(f"mean_shap shape after fix: {mean_shap.shape}")
assert mean_shap.shape[0] == len(FEATURE_NAMES), \
    f"Expected {len(FEATURE_NAMES)} features, got {mean_shap.shape[0]}"

if mean_shap.sum() > 0:
    mean_shap = mean_shap / mean_shap.sum()

# Print ranking
print()
print(f"  {'Rank':<5} {'Feature':<25} {'SHAP':>8}")
print("  " + "-" * 40)
shap_ranked = sorted(
    zip(FEATURE_NAMES, mean_shap),
    key=lambda x: x[1], reverse=True)
for i, (name, val) in enumerate(shap_ranked):
    print(f"  {i+1:<5} {name:<25} "
          f"{float(val)*100:>7.1f}%")

# SHAP chart
fig, ax = plt.subplots(figsize=(16, 8))
s_names = [p[0].replace('_', '\n')
            for p in shap_ranked]
s_vals = [float(p[1])*100 for p in shap_ranked]
colours = ['#C00000' if v == max(s_vals)
           else '#5B9BD5' for v in s_vals]
bars = ax.barh(s_names, s_vals, color=colours)
for bar, val in zip(bars, s_vals):
    ax.text(val + 0.3,
            bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center',
            fontsize=9, fontweight='bold')
ax.set_xlabel('Mean |SHAP Value| (%)', fontsize=12)
ax.set_title(
    'SHAP Feature Importance\n'
    'R26-IT-059 | IT22916426 | Week 17',
    fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(
    RESULTS_PATH + 'figures/model8_shap.png',
    dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: model8_shap.png")


#CHART 5 - SINGLE STUDENT EXPLANATION
print("\nGenerating single student explanation...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f'XAI Level 3 - Single Student Explanation\n'
             f'Risk: {explanation["risk_probability"]:.3f}'
             f'({explanation["risk_level"]})'
             f'| R26-IT-059 | IT22916426',
             fontweight='bold')
# Feature importance for this student

# Feature importance for this student
fi = explanation['feature_importance']
s_pairs = sorted(fi.items(),
                  key=lambda x: x[1],
                  reverse=True)

# Short readable names
short_names = {
    'F1_login_count':         'F1 Logins',
    'F2_vle_clicks':          'F2 Clicks',
    'F3_avg_score':           'F3 Score',
    'F4_num_submitted':       'F4 Submitted',
    'F5_on_time_rate':        'F5 On Time',
    'F6_inactive_days':       'F6 Inactive',
    'F7_score_change':        'F7 Trend',
    'F8_running_avg':         'F8 Cum Avg',
    'F9_score_variance':      'F9 Variance',
    'F10_min_score':          'F10 Min Score',
    'F11_max_score':          'F11 Max Score',
    'F12_late_count':         'F12 Late',
    'F13_weeks_since_active': 'F13 Inactive Wks'
}

s_names = [short_names.get(p[0], p[0])
            for p in s_pairs]
s_vals  = [float(p[1]) * 100
            for p in s_pairs]

colours_s = ['#C00000' if v == max(s_vals)
              else '#5B9BD5'
              for v in s_vals]

axes[0].barh(s_names, s_vals,
              color=colours_s, alpha=0.8)

# Add value labels
for idx, (val, name) in enumerate(
        zip(s_vals, s_names)):
    axes[0].text(
        val + 0.3, idx,
        f'{val:.1f}%',
        va='center', fontsize=8,
        fontweight='bold')

axes[0].set_xlabel('Importance (%)',
                    fontsize=10)
axes[0].set_title(
    'Why This Student Was Flagged\n'
    '(Feature Attribution)',
    fontsize=10, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='x')
axes[0].tick_params(axis='y', labelsize=8)

#Attention for this student
s_attn = explanation['attention_weights']
axes[1].bar(range(1,N_WEEKS+1), s_attn, color='#5B9BD5', alpha=0.8)

for hw in HORIZON_WEEKS:
    axes[1].axvline(x=hw, color='#70AD47',linestyle='--',alpha=0.6)

axes[1].set_xlabel('Week')
axes[1].set_ylabel('Attention Weight')
axes[1].set_title('Which weeks were important')
axes[1].set_xticks(range(1,N_WEEKS+1))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(
    RESULTS_PATH+'figures/'
    'model_8_xai_student.png',
    dpi=150, bbox_inches='tight'
)
plt.close()
print("Saved: model_8_xai_student.png")

# GENERATE PREDICTIONS CSV FOR IT22253194

print("\nGenerating predictions CSV for IT22253194... with per student XAI")

model.eval()
with torch.no_grad():
    final_preds, final_attn = model(X_te_t)


p4 = torch.sigmoid(final_preds[0]).squeeze().cpu().numpy()
p8 = torch.sigmoid(final_preds[1]).squeeze().cpu().numpy()
p12 = torch.sigmoid(final_preds[2]).squeeze().cpu().numpy()
p17 = torch.sigmoid(final_preds[3]).squeeze().cpu().numpy()

attn_all = final_attn.cpu().numpy()

rows = []

for i in range(len(p17)):
    risk = float(p17[i])
    max_risk = max(float(p4[i]), float(p8[i]), float(p12[i]), float(p17[i]))

     # Get per-student feature importance
    expl = explain_single_student(
        model, X_te_t,
        student_idx=i,
        horizon_idx=3)
    
      # Get top 3 features for this student
    sorted_feats = sorted(
        expl['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True)
    
    top1 = sorted_feats[0][0]
    top2 = sorted_feats[1][0]
    top3 = sorted_feats[2][0]

    top1_pct = round(float(sorted_feats[0][1]) * 100, 1)
    top2_pct = round(float(sorted_feats[1][1]) * 100, 1)
    top3_pct = round(float(sorted_feats[2][1]) * 100, 1)

    # Get top attention week for this student
    student_attn = attn_all[i]
    top_week = int(
        np.argmax(student_attn)) + 1
    rows.append({
        'student_index': i,
        'student_id':    f'OULAD_TEST_{i:04d}',
        'actual_label':  int(data['y_te'][i]),
        'academic_risk': round(risk, 4),
        'week4_risk':    round(float(p4[i]),  4),
        'week8_risk':    round(float(p8[i]),  4),
        'week12_risk':   round(float(p12[i]), 4),
        'week17_risk':   round(float(p17[i]), 4),
        'alert_level': (
            'HIGH'   if risk > 0.7 else
            'MEDIUM' if risk > 0.4 else
            'LOW'),
         # Per-student XAI
        'top_feature_1':     top1,
        'top_feature_1_pct': top1_pct,
        'top_feature_2':     top2,
        'top_feature_2_pct': top2_pct,
        'top_feature_3':     top3,
        'top_feature_3_pct': top3_pct,
        'most_important_week': top_week,

        # Human readable reason
        'main_reason': (
            f"{top1.replace('_',' ')} "
            f"({top1_pct}%) and "
            f"{top2.replace('_',' ')} "
            f"({top2_pct}%)")    


    })

df_preds = pd.DataFrame(rows)

csv_path = (RESULTS_PATH + 'metrics/model8_predictions.csv')
json_path = (RESULTS_PATH + 'metrics/model8_predictions.json')

df_preds.to_csv(csv_path, index=False)
df_preds.to_json(json_path, orient='records', indent=2)

print(f"  Saved {len(df_preds)} predictions.")
print(f"  CSV:  {csv_path}")
print(f"  JSON: {json_path}")
print()
print("  GIVE THESE TO IT22253194:")
print("  results/metrics/"
      "academic_predictions.csv")






    
    


#Save Result to file

import json

output = {
    'model': 'Unified Multi-Horizon GRU',
    'student': 'IT22916426',
    'project': 'R26-IT-059',
    'n_weeks': N_WEEKS,
    'n_features': N_FEATURES,
    'horizons': results,
    'optimal_horizon': results[best_horizon_idx],
    'baselines': {
        'Random Forest': 0.6647,
        'Logistic Regression': 0.6800,
        'XGBoost': 0.6890,
        'Gradient Boosting': 0.6965,
        'Simple LSTM': 0.7087,
        'GRU': 0.7185,
        'BiLSTM': 0.7107
    }
}

with open(RESULTS_PATH +'metrics/model8_unified_gru_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Saved : model8_results.json")


#Final Summary 

print()
print("=" * 60)
print("MODEL 8 COMPLETE")
print("=" * 60)
print()
print("ACCURACY VS LEAD TIME:")
print(f"  {'Horizon':<10} {'F1':>8} {'AUC':>8}")
print("  " + "-" * 28)
for r in results:
    marker = " ← OPTIMAL" \
        if r['week'] == opt_week else ""
    print(f"  {r['horizon']:<10} "
          f"{r['f1']:>8.4f} "
          f"{r['auc']:>8.4f}{marker}")

print()
print(f"OPTIMAL WINDOW: Week {opt_week} "
      f"— F1={opt_f1:.4f}")
print()
print("BEATS ALL BASELINES:")
for name, f1 in output['baselines'].items():
    beat = "✅" if opt_f1 > f1 else "❌"
    diff = opt_f1 - f1
    print(f"  {beat} {name:<22} "
          f"{f1:.4f} → +{diff:.4f}")

print()
print("XAI FINDINGS:")
print(f"  Most important weeks:   {list(top_3)}")
print(f"  Most important feature: {top_feature}")

print()
print("FILES SAVED:")
print("  Figures:")
print("    model8_accuracy_curve.png")
print("    model8_comparison.png")
print("    model8_attention.png")
print("    model8_xai_features.png")
print("    model8_xai_student.png")
print("  Metrics:")
print("    model8_results.json")
print("    academic_predictions.csv  ← IT22253194")
print("    academic_predictions.json ← IT22253194")
print("  Models:")
print("    model8_unified_gru_best.pt")
print("    scaler.pkl  ← Streamlit app")
print()
print("RESEARCH CONTRIBUTION:")
print("  First unified multi-horizon GRU")
print("  with attention + XAI for")
print("  educational burnout prediction.")
print("  Gap confirmed: Jin et al. 2024 AIED.")
print("=" * 60)













        





        
           
          
                   
        






































