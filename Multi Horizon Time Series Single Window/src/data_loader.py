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

    # 3-way-split
    #Step 1: crave out test

    X_temp, X_te, y_temp, y_te = train_test_split(
        X,y ,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=y
    )

    #Step 2: split remaining into train and val
    val_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_adjusted,
        random_state=SEED,
        stratify=y_temp
    )

    if verbose:
        n = len(X)
        print(f"Split:  Train={len(X_tr)} ({len(X_tr)/n*100:.0f}%) | "
              f"Val={len(X_val)} ({len(X_val)/n*100:.0f}%) | "
              f"Test={len(X_te)} ({len(X_te)/n*100:.0f}%)")
        print(f"At-risk: Train={y_tr.mean()*100:.1f}% | "
              f"Val={y_val.mean()*100:.1f}% | "
              f"Test={y_te.mean()*100:.1f}%")
        print()

    #Flat + scaled for ML models
    scaler = StandardScaler()
    X_flat_tr = scaler.fit_transform(X_tr.reshape(len(X_tr), -1))
    X_flat_val = scaler.transform(X_val.reshape(len(X_val), -1))
    X_flat_te = scaler.transform(X_te.reshape(len(X_te), -1))

    return {
        
        'X_tr': X_tr,  'X_val': X_val,  'X_te': X_te,
        'y_tr': y_tr,  'y_val': y_val,  'y_te': y_te,
        'X_flat_tr':  X_flat_tr,
        'X_flat_val': X_flat_val,
        'X_flat_te':  X_flat_te,
        'scaler': scaler
    }    

def print_metrics(name,f1,auc,precision,
                  recall,can_curve=False):
    """Print results in standard format."""

    print(f"\n{'='*52}")
    print(f" MODEL: {name}")
    print(f"{'='*52}")
    print(f"F1-score:   {f1:.4f}")
    print(f"AUC-ROC:        {auc:.4f}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f" Curve:          {'Yes' if can_curve else 'No'}")
    print(f"{'='*52}\n")


if name == "__main__":
    print("Testing data_loader.py...")
    data = get_data()  
    print("data_loader.py working correctly.")  




