"""
Train and evaluate the credit risk models.

Fits two models so results are comparable the way a real risk team would
present them: a logistic regression (still the industry baseline because
regulators can audit its coefficients) and a gradient-boosted tree model
(the accuracy upgrade most shops layer on top / challenger-champion it
against the regression).

Run with:  python -m src.train
Outputs:
  models/logreg.joblib
  models/xgb.joblib
  models/metrics.json
  reports/figures/roc_pr_curves.png
  reports/figures/ks_curve.png
  reports/figures/calibration.png
  reports/figures/shap_summary.png
  reports/figures/score_distribution.png
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_prep import MODEL_FEATURES, TARGET, prepare
from src.scorecard import pd_to_score, score_to_tier

MODELS_DIR = "models"
FIG_DIR = "reports/figures"
RANDOM_STATE = 42


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    good/bad distributions across score thresholds. Standard credit-risk
    discrimination metric alongside AUC/Gini."""
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score")
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    return float(np.max(np.abs(df["cum_bad"] - df["cum_good"])))


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = prepare()
    X = df[MODEL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # --- Logistic regression baseline (scaled features, class-balanced) --
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    )
    logreg.fit(X_train_scaled, y_train)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]

    # --- Gradient boosted model ------------------------------------------
    pos = y_train.sum()
    neg = len(y_train) - pos
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    # --- Metrics -----------------------------------------------------
    metrics = {}
    for name, proba in [("logistic_regression", logreg_proba), ("xgboost", xgb_proba)]:
        auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        ks = ks_statistic(y_test.values, proba)
        gini = 2 * auc - 1
        preds_at_50 = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds_at_50).ravel()
        metrics[name] = {
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "ks_statistic": round(ks, 4),
            "gini": round(gini, 4),
            "confusion_matrix_at_0.5": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    # Top-decile capture: of the riskiest 10% of applicants by predicted
    # PD, what share of actual defaults did we catch? This is the number a
    # collections/underwriting team actually cares about.
    order = np.argsort(-xgb_proba)
    top_decile_n = len(order) // 10
    top_decile_idx = order[:top_decile_n]
    captured = y_test.values[top_decile_idx].sum()
    metrics["xgboost"]["top_decile_capture_rate"] = round(
        float(captured / y_test.sum()), 4
    )

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump({"model": logreg, "scaler": scaler, "features": MODEL_FEATURES},
                os.path.join(MODELS_DIR, "logreg.joblib"))
    joblib.dump({"model": xgb, "features": MODEL_FEATURES},
                os.path.join(MODELS_DIR, "xgb.joblib"))

    print(json.dumps(metrics, indent=2))

    # --- Plots ---------------------------------------------------------
    _plot_roc_pr(y_test, {"Logistic Regression": logreg_proba, "XGBoost": xgb_proba})
    _plot_ks(y_test.values, xgb_proba)
    _plot_calibration(y_test.values, xgb_proba)
    _plot_score_distribution(y_test.values, xgb_proba)
    _plot_shap(xgb, X_test)

    print(f"\nSaved models to {MODELS_DIR}/, figures to {FIG_DIR}/")


def _plot_roc_pr(y_test, proba_dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, proba)
        axes[1].plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, proba):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "roc_pr_curves.png"), dpi=150)
    plt.close(fig)


def _plot_ks(y_test, proba):
    df = pd.DataFrame({"y": y_test, "score": proba}).sort_values("score").reset_index(drop=True)
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    df["pct"] = (df.index + 1) / len(df)
    ks_idx = (df["cum_bad"] - df["cum_good"]).abs().idxmax()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(df["pct"], df["cum_good"], label="Cumulative % Good")
    ax.plot(df["pct"], df["cum_bad"], label="Cumulative % Bad")
    ax.vlines(df.loc[ks_idx, "pct"], df.loc[ks_idx, "cum_good"], df.loc[ks_idx, "cum_bad"],
               color="black", linestyle="--",
               label=f"KS = {abs(df.loc[ks_idx, 'cum_bad'] - df.loc[ks_idx, 'cum_good']):.3f}")
    ax.set_xlabel("Population percentile (sorted by predicted PD)")
    ax.set_ylabel("Cumulative distribution")
    ax.set_title("KS Curve (XGBoost)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ks_curve.png"), dpi=150)
    plt.close(fig)


def _plot_calibration(y_test, proba):
    from sklearn.calibration import calibration_curve

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(mean_pred, frac_pos, marker="o", label="XGBoost")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted PD (bin)")
    ax.set_ylabel("Observed default rate (bin)")
    ax.set_title("Calibration Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "calibration.png"), dpi=150)
    plt.close(fig)


def _plot_score_distribution(y_test, proba):
    scores = pd_to_score(proba)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(scores[y_test == 0], bins=40, alpha=0.6, label="Non-defaulters", density=True)
    ax.hist(scores[y_test == 1], bins=40, alpha=0.6, label="Defaulters", density=True)
    ax.set_xlabel("Scaled credit score (300-850)")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Outcome")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "score_distribution.png"), dpi=150)
    plt.close(fig)


def _plot_shap(xgb, X_test):
    import shap

    sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(sample)
    fig = plt.figure(figsize=(7, 5.5))
    shap.summary_plot(shap_values, sample, show=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
