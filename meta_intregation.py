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


