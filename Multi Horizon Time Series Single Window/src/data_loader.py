import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Configuration

SEED = 42 
PROCESSED_PATH = "../data/processed/"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

np.random.seed(SEED)

def get_data(verbose=True):

    #Load process data 
    try:
        X = np.load(PROCESSED_PATH + "X.npy")
        y = np.load(PROCESSED_PATH + "y.npy")


    except FileNotFoundError:
        raise FileNotFoundError(
             "X_sequences.npy or y_labels.npy not found.\n"
            "Please run Notebook 02_feature_engineering.ipynb first."
        ) 
    
    if verbose: 
        print(f"Loaded  X: {X.shape}  (students x weeks x features)")
        print(f"Loaded  y: {y.shape}")
        print(f"At-risk:   {y.mean()*100:.1f}%")
        print()

