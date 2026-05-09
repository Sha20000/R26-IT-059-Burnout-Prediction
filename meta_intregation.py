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
