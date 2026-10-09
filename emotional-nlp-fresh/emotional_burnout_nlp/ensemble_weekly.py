"""
Stage 4 — BERT + RoBERTa soft-voting ensemble on the weekly dataset v2, and a
comparison of every model trained on the same student-level split.
IT22196392 — Induwara K.P.Y. | R26-IT-059

The ensemble weight is chosen on validation macro-F1 only; the test set is scored once.
Significance: paired student-level bootstrap of the macro-F1 difference between models.

Run:  .venv/Scripts/python.exe ensemble_weekly.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from preprocess_weekly import ID2LABEL
from train_baseline_weekly import FIG_DIR, LABELS, full_report, scores

BASE_DIR = Path(__file__).parent
PRED_DIR = BASE_DIR / "results" / "predictions"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PROB_COLS = [f"p_{l}" for l in LABELS]
SEED = 42


def load(model, split):
    return pd.read_csv(PRED_DIR / f"weekly_v2_{model}_{split}_predictions.csv", dtype={"student_id": str})


def paired_bootstrap(df, pred_a, pred_b, n=1000, seed=SEED):
    """Macro-F1(a) - macro-F1(b), resampling whole students. Returns mean diff, 95% CI,
    and the share of resamples where a is not better (a one-sided p-value)."""
    rng = np.random.default_rng(seed)
    rows = df.groupby("student_id").indices
    students = np.array(list(rows))
    y = df["label_id"].to_numpy()
    diffs = []
    for _ in range(n):
        idx = np.concatenate([rows[s] for s in rng.choice(students, len(students), replace=True)])
        diffs.append(f1_score(y[idx], pred_a[idx], average="macro") - f1_score(y[idx], pred_b[idx], average="macro"))
    diffs = np.array(diffs)
    return {"mean_diff": round(float(diffs.mean()), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "p_value_one_sided": round(float((diffs <= 0).mean()), 4)}


def main():
    val = {m: load(m, "val") for m in ["bert", "roberta"]}
    test = {m: load(m, "test") for m in ["bert", "roberta"]}
    for d in (val, test):                                    # rows must line up between models
        assert (d["bert"][["student_id", "week"]].to_numpy() == d["roberta"][["student_id", "week"]].to_numpy()).all()
    y_val = val["roberta"]["label_id"].to_numpy()

    # 1. Weight search on validation -------------------------------------------------------
    grid = []
    for w in np.round(np.arange(0, 1.0001, 0.05), 2):
        p = w * val["roberta"][PROB_COLS].to_numpy() + (1 - w) * val["bert"][PROB_COLS].to_numpy()
        grid.append((float(w), f1_score(y_val, p.argmax(1), average="macro")))
    best_f1 = max(f for _, f in grid)
    w_rob = min((w for w, f in grid if f == best_f1), key=lambda w: abs(w - 0.5))   # ties -> closest to equal mix
    print(f"[1] Best RoBERTa weight on validation: {w_rob} (BERT {1 - w_rob:.2f}), val macro-F1 {best_f1:.4f}")

    # 2. Test ------------------------------------------------------------------------------
    t = test["roberta"]
    probs = w_rob * t[PROB_COLS].to_numpy() + (1 - w_rob) * test["bert"][PROB_COLS].to_numpy()
    ens_pred = probs.argmax(1)
    report = full_report(t, ens_pred)

    out = t[["student_id", "week", "status", "label_id"]].copy()
    out["pred_id"] = ens_pred
    out["pred_label"] = out["pred_id"].map(ID2LABEL)
    out[PROB_COLS] = probs.round(5)
    out.to_csv(PRED_DIR / "weekly_v2_ensemble_test_predictions.csv", index=False)

    # 3. All models on the same test set ---------------------------------------------------
    base = json.loads((METRICS_DIR / "weekly_v2_baseline_results.json").read_text())
    base_pred = pd.read_csv(PRED_DIR / "weekly_v2_baseline_test_predictions.csv", dtype={"student_id": str})
    assert (base_pred[["student_id", "week"]].to_numpy() == t[["student_id", "week"]].to_numpy()).all()
    preds = {f"{base['best_model']} (TF-IDF)": base_pred["pred_id"].to_numpy(),
             "BERT": test["bert"]["pred_id"].to_numpy(),
             "RoBERTa": t["pred_id"].to_numpy(),
             f"Ensemble (RoBERTa {w_rob:.2f} + BERT {1 - w_rob:.2f})": ens_pred}
    table = {name: full_report(t, p) for name, p in preds.items()}
    names = list(preds)
    significance = {
        "RoBERTa vs " + names[0]: paired_bootstrap(t, preds["RoBERTa"], preds[names[0]]),
        "RoBERTa vs BERT": paired_bootstrap(t, preds["RoBERTa"], preds["BERT"]),
        "Ensemble vs RoBERTa": paired_bootstrap(t, ens_pred, preds["RoBERTa"]),
    }

    print(f"\n{'model':42s} {'acc':>6s} {'macroF1':>8s} {'95% CI':>17s} {'wF1':>6s} {'riskAcc':>8s} {'atRiskRec':>9s}")
    for n, r in table.items():
        print(f"{n:42s} {r['accuracy']:6.4f} {r['f1_macro']:8.4f} {str(r['f1_macro_95ci_student_bootstrap']):>17s} "
              f"{r['f1_weighted']:6.4f} {r['risk_level']['accuracy']:8.4f} {r['at_risk_binary']['recall']:9.4f}")
    print("\nPer-week macro-F1:")
    for n, r in table.items():
        print(f"  {n:42s}", {w: v["f1_macro"] for w, v in r["per_week"].items()})
    print("\nPaired bootstrap (macro-F1 difference):")
    for k, v in significance.items():
        print(f"  {k:40s} {v}")

    METRICS_DIR.joinpath("weekly_v2_ensemble_results.json").write_text(json.dumps({
        "researcher": "IT22196392 - Induwara K.P.Y.",
        "dataset": "Combined_Data_all_weeks_v2 (student-level split, see data/processed/weekly_v2)",
        "ensemble": {"method": "weighted soft voting of class probabilities",
                     "roberta_weight": w_rob, "bert_weight": round(1 - w_rob, 2),
                     "chosen_on": "validation macro-F1", "val_f1_macro": round(best_f1, 4),
                     "weight_grid_val_f1": {str(w): round(f, 4) for w, f in grid}},
        "test": report, "comparison_test": table, "significance_paired_student_bootstrap": significance}, indent=2))

    # 4. Figures ---------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short = [n.split(" (")[0] if not n.startswith("Ensemble") else "Ensemble" for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    x = np.arange(len(names))
    for k, (metric, label) in enumerate([("accuracy", "Accuracy"), ("f1_weighted", "Weighted F1"), ("f1_macro", "Macro F1")]):
        vals = [table[n][metric] for n in names]
        bars = axes[0].bar(x + (k - 1) * 0.27, vals, 0.27, label=label)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=12 if metric == "f1_macro" else 2)
    ci = np.array([table[n]["f1_macro_95ci_student_bootstrap"] for n in names])
    mf1 = np.array([table[n]["f1_macro"] for n in names])
    axes[0].errorbar(x + 0.27, mf1, yerr=[mf1 - ci[:, 0], ci[:, 1] - mf1], fmt="none", ecolor="k", capsize=3, lw=1)
    axes[0].set_xticks(x, short); axes[0].set_ylim(0.6, 1.0)
    axes[0].set_title("Test set (error bars: 95% CI, student bootstrap)"); axes[0].legend(fontsize=8, loc="lower right")
    weeks = sorted(table[names[0]]["per_week"])
    for n, s in zip(names, short):
        axes[1].plot(weeks, [table[n]["per_week"][w]["f1_macro"] for w in weeks], marker="o", label=s)
    axes[1].set_xticks(weeks); axes[1].set_xlabel("Semester week"); axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1 by semester week — test set"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.suptitle("Weekly dataset v2 — all models | IT22196392", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_v2_all_models_comparison.png", dpi=150)
    plt.close(fig)

    cm = np.array(report["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Ensemble — test confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_v2_ensemble_confusion.png", dpi=150)
    plt.close(fig)
    print("\n[4] Saved results/metrics/weekly_v2_ensemble_results.json and figures")


if __name__ == "__main__":
    main()
