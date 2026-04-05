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




