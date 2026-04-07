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



