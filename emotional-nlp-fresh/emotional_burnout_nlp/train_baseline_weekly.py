"""
Stage 2 — Classical baseline models on the weekly dataset v2.
IT22196392 — Induwara K.P.Y. | R26-IT-059

Uses the student-level 70/15/15 split from preprocess_weekly.py, so the numbers are
directly comparable with the BERT/RoBERTa models trained later on the same split.

Models (TF-IDF word 1-2 grams + char 3-5 grams, class-weighted):
  0. Majority class (floor)
  1. Logistic Regression     — C tuned on validation
  2. Linear SVM              — C tuned on validation
  3. Complement Naive Bayes  — alpha tuned on validation
  4. Random Forest           — same settings as the old baseline notebook
Selection metric: validation macro-F1. The test set is scored once, at the end.

Run:  .venv/Scripts/python.exe train_baseline_weekly.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from preprocess_weekly import ID2LABEL, LABEL2ID, OUT_DIR as DATA_DIR

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models" / "saved" / "baseline_weekly_v2"
METRICS_PATH = BASE_DIR / "results" / "metrics" / "weekly_v2_baseline_results.json"
PRED_DIR = BASE_DIR / "results" / "predictions"
FIG_DIR = BASE_DIR / "results" / "figures"
SEED = 42
LABELS = [ID2LABEL[i] for i in range(len(ID2LABEL))]

# Same risk levels as the dashboard (app.py / serve_model.py).
RISK = {"Normal": "Low", "Stress": "Medium", "Anxiety": "Medium", "Depression": "High",
        "Bipolar": "High", "Personality disorder": "High", "Suicidal": "Critical"}


class TextFeatures:
    """Word + character TF-IDF. Stop words are kept on purpose: 'not', 'no', 'never'
    carry meaning in mental-health text."""

    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=100_000,
                                    sublinear_tf=True, strip_accents="unicode")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=150_000,
                                    sublinear_tf=True, strip_accents="unicode")

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


def scores(y_true, y_pred):
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4)}


def student_bootstrap_ci(df, y_pred, n=1000, seed=SEED):
    """95% CI for test macro-F1, resampling whole students (their weeks are not independent)."""
    rng = np.random.default_rng(seed)
    rows_by_student = df.groupby("student_id").indices
    students = np.array(list(rows_by_student))
    y_true = df["label_id"].to_numpy()
    vals = []
    for _ in range(n):
        pick = rng.choice(students, size=len(students), replace=True)
        idx = np.concatenate([rows_by_student[s] for s in pick])
        vals.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def full_report(df, y_pred):
    y_true = df["label_id"].to_numpy()
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    out = scores(y_true, y_pred)
    out["f1_macro_95ci_student_bootstrap"] = student_bootstrap_ci(df, y_pred)
    out["per_class"] = {LABELS[i]: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                                    "f1": round(float(f[i]), 4), "support": int(s[i])} for i in range(len(LABELS))}
    out["per_week"] = {int(w): scores(y_true[m], y_pred[m])
                       for w in sorted(df["week"].unique()) for m in [(df["week"] == w).to_numpy()]}
    # Dashboard risk level (Low / Medium / High / Critical)
    risk_true = [RISK[ID2LABEL[i]] for i in y_true]
    risk_pred = [RISK[ID2LABEL[i]] for i in y_pred]
    out["risk_level"] = {"accuracy": round(accuracy_score(risk_true, risk_pred), 4),
                         "f1_macro": round(f1_score(risk_true, risk_pred, average="macro"), 4)}
    # Early-warning view: any non-Normal label = "at risk"
    at_true, at_pred = y_true != LABEL2ID["Normal"], y_pred != LABEL2ID["Normal"]
    bp, br, bf, _ = precision_recall_fscore_support(at_true, at_pred, average="binary")
    out["at_risk_binary"] = {"precision": round(float(bp), 4), "recall": round(float(br), 4), "f1": round(float(bf), 4)}
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist()
    return out


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    train, val, test = (pd.read_csv(DATA_DIR / f"{s}.csv", dtype={"student_id": str}) for s in ["train", "val", "test"])
    print(f"[1] train {len(train):,} | val {len(val):,} | test {len(test):,}")

    t0 = time.time()
    feats = TextFeatures()
    X_tr = feats.fit_transform(train["text_clean"])
    X_va, X_te = feats.transform(val["text_clean"]), feats.transform(test["text_clean"])
    y_tr, y_va = train["label_id"].to_numpy(), val["label_id"].to_numpy()
    print(f"[2] TF-IDF features: {X_tr.shape[1]:,} ({time.time() - t0:.0f}s)")

    candidates = {
        "Majority class": [({}, lambda: DummyClassifier(strategy="most_frequent"))],
        "Logistic Regression": [({"C": c}, lambda c=c: LogisticRegression(
            C=c, class_weight="balanced", max_iter=3000, random_state=SEED)) for c in [1, 4, 16]],
        "Linear SVM": [({"C": c}, lambda c=c: LinearSVC(
            C=c, class_weight="balanced", max_iter=5000, random_state=SEED)) for c in [0.1, 0.3, 1.0]],
        "Complement NB": [({"alpha": a}, lambda a=a: ComplementNB(alpha=a)) for a in [0.1, 0.3, 1.0]],
        "Random Forest": [({"n_estimators": 300}, lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))],
    }

    results, fitted = {}, {}
    for name, grid in candidates.items():
        best = None
        for params, make in grid:
            t0 = time.time()
            clf = make().fit(X_tr, y_tr)
            val_f1 = f1_score(y_va, clf.predict(X_va), average="macro")
            print(f"[3] {name:20s} {str(params):22s} val macro-F1 {val_f1:.4f} ({time.time() - t0:.0f}s)")
            if best is None or val_f1 > best[0]:
                best = (val_f1, params, clf)
        val_f1, params, clf = best
        fitted[name] = clf
        results[name] = {"best_params": params, "val_f1_macro": round(val_f1, 4),
                         "val": scores(y_va, clf.predict(X_va))}

    # Test — scored once per model, after all choices were made on validation.
    for name, clf in fitted.items():
        results[name]["test"] = full_report(test, clf.predict(X_te))

    best_name = max((n for n in results if n != "Majority class"), key=lambda n: results[n]["val_f1_macro"])
    best = fitted[best_name]
    print(f"\n[4] Best on validation: {best_name} {results[best_name]['best_params']}")
    print(f"{'model':22s} {'val mF1':>8s} {'test acc':>9s} {'test mF1':>9s} {'test wF1':>9s}")
    for n, r in results.items():
        t = r["test"]
        print(f"{n:22s} {r['val_f1_macro']:8.4f} {t['accuracy']:9.4f} {t['f1_macro']:9.4f} {t['f1_weighted']:9.4f}")
    tb = results[best_name]["test"]
    print(f"\nBest test macro-F1 {tb['f1_macro']} (95% CI {tb['f1_macro_95ci_student_bootstrap']}), "
          f"per week: { {w: v['f1_macro'] for w, v in tb['per_week'].items()} }")
    print(classification_report(test["label_id"], best.predict(X_te), target_names=LABELS, digits=4))

    # Save -----------------------------------------------------------------------------
    joblib.dump({"features": feats, "model": best, "label2id": LABEL2ID, "name": best_name},
                MODEL_DIR / "best_baseline.joblib", compress=3)
    out = {"dataset": "Combined_Data_all_weeks_v2 (student-level split, see data/processed/weekly_v2)",
           "researcher": "IT22196392 - Induwara K.P.Y.",
           "features": f"TF-IDF word 1-2 grams + char_wb 3-5 grams, {X_tr.shape[1]} features",
           "selection_metric": "validation macro-F1", "best_model": best_name, "models": results}
    METRICS_PATH.write_text(json.dumps(out, indent=2))

    pred = test[["student_id", "week", "status", "label_id"]].copy()
    pred["pred_id"] = best.predict(X_te)
    pred["pred_label"] = pred["pred_id"].map(ID2LABEL)
    if hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_te)
        for i, lab in enumerate(LABELS):
            pred[f"p_{lab}"] = proba[:, i].round(5)
    pred.to_csv(PRED_DIR / "weekly_v2_baseline_test_predictions.csv", index=False)

    # Figures --------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(names))
    for k, (metric, label) in enumerate([("accuracy", "Accuracy"), ("f1_weighted", "Weighted F1"), ("f1_macro", "Macro F1")]):
        vals = [results[n]["test"][metric] for n in names]
        bars = axes[0].bar(x + (k - 1) * 0.27, vals, 0.27, label=label)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    axes[0].set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=9)
    axes[0].set_ylim(0, 1.05); axes[0].set_title("Baselines — test set"); axes[0].legend(fontsize=8)
    weeks = sorted(tb["per_week"])
    for n in names[1:]:
        axes[1].plot(weeks, [results[n]["test"]["per_week"][w]["f1_macro"] for w in weeks], marker="o", label=n)
    axes[1].set_xticks(weeks); axes[1].set_xlabel("Semester week"); axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Macro F1 by week — test set"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.suptitle("Weekly dataset v2 — classical baselines | IT22196392", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_v2_baseline_comparison.png", dpi=150)
    plt.close(fig)

    cm = np.array(tb["confusion_matrix"], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=40, ha="right"); ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{best_name} — test confusion matrix (row-normalised)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "weekly_v2_baseline_confusion.png", dpi=150)
    plt.close(fig)

    print(f"[5] Saved model -> {MODEL_DIR.relative_to(BASE_DIR)}, metrics -> {METRICS_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
