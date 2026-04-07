"""
data_loader.py
==============
R26-IT-059 | IT22916426 | Mahavitha S.M.

PURPOSE:
    Load processed OULAD features and split into
    Train (70%) / Validation (15%) / Test (15%)

    All model files import this file.
    Run this file directly to test it works.

USAGE:
    from data_loader import get_data, print_metrics
    data = get_data()
    X_tr = data['X_tr']   # training features
    y_tr = data['y_tr']   # training labels
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Settings ──────────────────────────────────────────────
SEED           = 42
PROCESSED_PATH = '../data/processed/'
TRAIN_RATIO    = 0.70   # 70% for training
VAL_RATIO      = 0.15   # 15% for validation
TEST_RATIO     = 0.15   # 15% for final test

np.random.seed(SEED)


def get_data(verbose=True):
    """
    Load X and y from disk.
    Split into train / validation / test.
    Normalise features.
    Return everything in a dictionary.
    """

    # ── Step 1: Load saved numpy files ────────────────────
    try:
        X = np.load(PROCESSED_PATH + 'X_sequences.npy')
        y = np.load(PROCESSED_PATH + 'y_labels.npy')
    except FileNotFoundError:
        raise FileNotFoundError(
            "\nFiles not found. "
            "Please run notebook 02 first.\n"
            f"Looking in: {PROCESSED_PATH}")

    n_students, n_weeks, n_features = X.shape

    if verbose:
        print(f"Loaded  X: {X.shape}  "
              f"(students x weeks x features)")
        print(f"Loaded  y: {y.shape}")
        print(f"At-risk:   {y.mean()*100:.1f}%")
        print()

    # ── Step 2: Split data into 3 parts ───────────────────
    # IMPORTANT: Split BEFORE normalising
    # to prevent data leakage from test into train

    # First carve out test set
    X_temp, X_te, y_temp, y_te = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=y)          # keep same at-risk ratio

    # Then split remaining into train + validation
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=SEED,
        stratify=y_temp)

    if verbose:
        n = len(X)
        print(f"Split:  Train={len(X_tr)} "
              f"({len(X_tr)/n*100:.0f}%) | "
              f"Val={len(X_val)} "
              f"({len(X_val)/n*100:.0f}%) | "
              f"Test={len(X_te)} "
              f"({len(X_te)/n*100:.0f}%)")
        print(f"At-risk: "
              f"Train={y_tr.mean()*100:.1f}% | "
              f"Val={y_val.mean()*100:.1f}% | "
              f"Test={y_te.mean()*100:.1f}%")
        print()

    # ── Step 3: Normalise features ─────────────────────────
    # Flatten 3D to 2D for scaling
    # Fit scaler on TRAIN only
    # Then apply to val and test

    scaler = StandardScaler()

    # Reshape: (students, weeks, features) → (students*weeks, features)
    X_tr_flat  = X_tr.reshape(len(X_tr), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_te_flat  = X_te.reshape(len(X_te), -1)

    # Fit on train, transform all
    X_tr_scaled  = scaler.fit_transform(X_tr_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_te_scaled  = scaler.transform(X_te_flat)

    # Reshape back to 3D for DL models
    X_tr  = X_tr_scaled.reshape(
        len(X_tr), n_weeks, n_features).astype(np.float32)
    X_val = X_val_scaled.reshape(
        len(X_val), n_weeks, n_features).astype(np.float32)
    X_te  = X_te_scaled.reshape(
        len(X_te), n_weeks, n_features).astype(np.float32)

    # Keep flat versions for ML models (model1-4)
    X_flat_tr  = X_tr.reshape(len(X_tr), -1)
    X_flat_val = X_val.reshape(len(X_val), -1)
    X_flat_te  = X_te.reshape(len(X_te), -1)

    # ── Return everything ──────────────────────────────────
    return {
        # 3D arrays for LSTM/GRU/BiLSTM models
        'X_tr':  X_tr,   'X_val':  X_val,   'X_te':  X_te,
        # Labels
        'y_tr':  y_tr,   'y_val':  y_val,   'y_te':  y_te,
        # 2D flat arrays for ML models
        'X_flat_tr':  X_flat_tr,
        'X_flat_val': X_flat_val,
        'X_flat_te':  X_flat_te,
        # Scaler (in case you need it later)
        'scaler': scaler
    }


def get_pos_weight(y_train):
    """
    Calculate class weight for imbalanced data.
    At-risk class gets more weight so model
    does not ignore minority class.
    """
    n_safe    = (y_train == 0).sum()
    n_atrisk  = (y_train == 1).sum()
    weight    = n_safe / n_atrisk
    return float(weight)


def find_best_threshold(model, X_val, y_val,
                        device, is_torch=True):
    """
    Find the best prediction threshold on validation set.
    Default 0.5 is not always best for imbalanced data.
    """
    import torch
    from sklearn.metrics import f1_score

    if is_torch:
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.FloatTensor(X_val).to(device))
            ).squeeze().cpu().numpy()
    else:
        probs = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.5
    best_f1     = 0

    for thresh in np.arange(0.2, 0.71, 0.05):
        preds = (probs > thresh).astype(int)
        score = f1_score(y_val, preds,
                         zero_division=0)
        if score > best_f1:
            best_f1     = score
            best_thresh = thresh

    return best_thresh, best_f1


def print_metrics(name, f1, auc,
                  precision, recall,
                  can_curve=False):
    """Print evaluation results in standard format."""
    print(f"\n{'='*52}")
    print(f"  MODEL:     {name}")
    print(f"{'='*52}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Curve:     "
          f"{'YES ✓' if can_curve else 'NO  ✗'}")
    print(f"{'='*52}")


# ── Test this file directly ────────────────────────────────
if __name__ == '__main__':
    print("Testing data_loader.py...")
    data = get_data()
    weight = get_pos_weight(data['y_tr'])
    print(f"Class weight for at-risk: {weight:.2f}x")
    print("data_loader.py working correctly.")