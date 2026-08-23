"""
mcnemar_significance_test.py

Tests whether the Week-17 F1 improvement of the Unified Multi-Horizon
GRU (0.7227) over the single-task GRU baseline (0.7185) is statistically
significant, using McNemar's test on paired per-student predictions.

HOW TO RUN:
    Place this file in your project's `src/` folder (same folder as
    data_loader.py, model_8_unified_gru.py, model_6_gru.py, etc.)
    Then run:
        python mcnemar_significance_test.py

REQUIREMENTS:
    pip install statsmodels --break-system-packages   (if not installed)

WHAT THIS SCRIPT DOES:
    1. Loads your test set (X_te, y_te) via data_loader.py
    2. Loads BOTH saved model checkpoints
    3. Runs inference with BOTH models on the IDENTICAL test set
    4. Extracts each model's Week 17 prediction for every student
    5. Runs McNemar's test comparing the two sets of predictions
    6. Prints the contingency table, p-value, and a plain-English verdict
"""

import sys, os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
from statsmodels.stats.contingency_tables import mcnemar
from data_loader import get_data


# ============================================================
# STEP 1 — EDIT THESE PATHS to match your actual saved files
# ============================================================
PROPOSED_MODEL_CHECKPOINT = "../models/saved/model8_unified_gru_best.pt"   # <-- your Model 8 checkpoint filename
BASELINE_MODEL_CHECKPOINT = "../models/saved/model6_gru_best.pt"            # <-- your single-task GRU checkpoint filename

PROPOSED_MODEL_THRESHOLD = 0.55   # <-- Week 17 threshold from Table 3.4 / your JSON results
BASELINE_MODEL_THRESHOLD = 0.50   # <-- single-task GRU's own optimised threshold (check your saved results)


def run_mcnemar_test(y_true, preds_a, preds_b, name_a="Proposed Model", name_b="Baseline"):
    """Builds the 2x2 contingency table and runs McNemar's test."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    n00 = np.sum(correct_a & correct_b)             # both correct
    n01 = np.sum(correct_a & ~correct_b)             # A right, B wrong
    n10 = np.sum(~correct_a & correct_b)             # A wrong, B right
    n11 = np.sum(~correct_a & ~correct_b)            # both wrong

    print(f"\nContingency table ({name_a} vs {name_b}):")
    print(f"  Both correct:                    {n00}")
    print(f"  {name_a} right, {name_b} wrong:   {n01}")
    print(f"  {name_a} wrong, {name_b} right:   {n10}")
    print(f"  Both wrong:                      {n11}")
    print(f"  Total disagreements (n01+n10):   {n01 + n10}")

    table = [[n00, n01], [n10, n11]]
    use_exact = (n01 + n10) < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    print(f"\nMcNemar's test ({'exact binomial' if use_exact else 'chi-square, continuity-corrected'}):")
    print(f"  statistic = {result.statistic:.4f}")
    print(f"  p-value   = {result.pvalue:.4f}")

    if result.pvalue < 0.05:
        print(f"\n  RESULT: p < 0.05 -> the difference IS statistically significant.")
    else:
        print(f"\n  RESULT: p >= 0.05 -> the difference is NOT statistically significant")
        print(f"          at the 5% level; the F1 improvement may be within random variation.")

    return result, (n00, n01, n10, n11)


def get_predictions_from_checkpoint(checkpoint_path, model, X_te_t, week17_output_index, threshold):
    """
    Loads a checkpoint into `model`, runs inference on X_te_t, and
    returns binary (0/1) predictions at the Week 17 horizon.

    `week17_output_index`:
        - For Model 8 (4 outputs): index 3 (0=Wk4, 1=Wk8, 2=Wk12, 3=Wk17)
        - For single-task GRU (1 output): index 0
    """
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        outputs = model(X_te_t)
        # If model returns a list (Model 8's multi-horizon output):
        if isinstance(outputs, (list, tuple)):
            logits = outputs[0][week17_output_index] if isinstance(outputs[0], (list, tuple)) else outputs[week17_output_index]
        else:
            logits = outputs
        probs = torch.sigmoid(logits).squeeze().numpy()
    preds = (probs >= threshold).astype(int)
    return preds


if __name__ == "__main__":
    print("=" * 60)
    print("McNemar's Test: Proposed Model vs Single-task GRU")
    print("=" * 60)

    # ------------------------------------------------------
    # STEP 2 — Load test data (same for both models)
    # ------------------------------------------------------
    data = get_data()
    y_te = data['y_te'].astype(int)
    X_te_t = torch.FloatTensor(data['X_te'])

    print(f"\nTest set size: {len(y_te)} students")

    # ------------------------------------------------------
    # STEP 3 — Load both models and get predictions
    # ------------------------------------------------------
    # CONFIRMED for UnifiedMultiHorizonGRU (Proposed Model):
    #   - forward() returns (predictions, attn_weights)
    #   - predictions is a LIST of 4 tensors, one per horizon
    #   - Week 17 is predictions[3] (last in horizon_indices)
    #   - __init__ needs: n_features, hidden, num_layers,
    #     dropout, horizon_indices (these have defaults tied
    #     to constants HIDDEN_SIZE/NUM_LAYERS/DROPOUT/
    #     HORIZON_INDICES defined in model_8_unified_gru.py —
    #     import those too so the defaults resolve correctly)

    from model_8_unified_gru import UnifiedMultiHorizonGRU
    from model_6_gru import SimpleGRU   # confirmed class name: SimpleGRU

    proposed_model = UnifiedMultiHorizonGRU(n_features=13)
    proposed_model.load_state_dict(
        torch.load(PROPOSED_MODEL_CHECKPOINT, map_location="cpu")
    )
    proposed_model.eval()
    with torch.no_grad():
        predictions_list, _ = proposed_model(X_te_t)
        # predictions_list[3] = Week 17's raw logits, shape (batch, 1)
        week17_logits = predictions_list[3].squeeze()
        week17_probs = torch.sigmoid(week17_logits).numpy()
    proposed_preds = (week17_probs >= PROPOSED_MODEL_THRESHOLD).astype(int)

    # CONFIRMED for SimpleGRU (Baseline):
    #   - forward() returns a SINGLE tensor, shape (batch, 1)
    #   - Uses gru_out[:, -1, :] — the LAST timestep (Week 17,
    #     since input is the full 17-week sequence), so NO
    #     index selection needed like Model 8's predictions[3]
    #   - No attention, no horizon_indices — much simpler

    baseline_model = SimpleGRU(n_features=13)
    baseline_model.load_state_dict(
        torch.load(BASELINE_MODEL_CHECKPOINT, map_location="cpu")
    )
    baseline_model.eval()
    with torch.no_grad():
        baseline_logits = baseline_model(X_te_t).squeeze()
        baseline_probs = torch.sigmoid(baseline_logits).numpy()
    baseline_preds = (baseline_probs >= BASELINE_MODEL_THRESHOLD).astype(int)
    # ------------------------------------------------------
    # STEP 4 — Run the test
    # ------------------------------------------------------
    run_mcnemar_test(y_te, proposed_preds, baseline_preds,
                      name_a="Unified Multi-Horizon GRU",
                      name_b="Single-task GRU")