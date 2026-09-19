"""
Lending Club: PD model, benchmark against LC's grade, business layer, macro
stress test, and drift monitoring.

    python -m src.lc_data      # build data/processed/lending_club.parquet
    python -m src.lc_train     # everything below

Design
  Train: 36-month loans issued 2007-2014
  Test:  36-month loans issued 2015 (out of time, as a lender would validate)
  Drift: 2016-2018 applications (outcomes not final yet, so unlabeled)

Outputs
  models/lending_club_xgb.joblib
  reports/lending_club/metrics.json
  reports/lending_club/cutoff_table.csv
  reports/lending_club/strategies.csv
  reports/lending_club/expected_vs_realized_loss.csv
  reports/lending_club/stress_scenarios.csv
  reports/lending_club/psi.csv
  reports/figures/lc_*.png
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
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src import business, macro, monitoring
from src.lc_data import (
    MACRO_FEATURES,
    MODEL_FEATURES,
    PROCESSED_PATH,
    TARGET,
    prepare,
    to_model_frame,
)
from src.train import ks_statistic

MODEL_PATH = "models/lending_club_xgb.joblib"
REPORT_DIR = "reports/lending_club"
FIG_DIR = "reports/figures"
RANDOM_STATE = 42


def load() -> pd.DataFrame:
    if os.path.exists(PROCESSED_PATH):
        return pd.read_parquet(PROCESSED_PATH)
    return prepare()


def fit_model(train: pd.DataFrame, features: list[str] = MODEL_FEATURES) -> XGBClassifier:
    fit, early = train_test_split(
        train, test_size=0.1, stratify=train[TARGET], random_state=RANDOM_STATE
    )
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=50,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        early_stopping_rounds=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(
        to_model_frame(fit, features), fit[TARGET],
        eval_set=[(to_model_frame(early, features), early[TARGET])],
        verbose=False,
    )
    return model


def sub_grade_rank(sub_grade: pd.Series) -> np.ndarray:
    """A1 -> 1 ... G5 -> 35, so LC's grade can be scored like a model."""
    return (sub_grade.str[0].map(ord) - ord("A")) * 5 + sub_grade.str[1].astype(int)


def main() -> None:
    for d in (REPORT_DIR, FIG_DIR, os.path.dirname(MODEL_PATH)):
        os.makedirs(d, exist_ok=True)

    df = load()
    labeled = df[df[TARGET].notna()].copy()
    train = labeled[labeled["issue_year"] <= 2014].copy()
    test = labeled[labeled["issue_year"] == 2015].copy()
    drift = df[df["issue_year"] >= 2016].copy()
    print(f"train {len(train):,}  test {len(test):,}  drift {len(drift):,}")

    # --- PD model ------------------------------------------------------
    model = fit_model(train)
    test["pd"] = model.predict_proba(to_model_frame(test))[:, 1]
    y = test[TARGET].to_numpy()

    grade_rank = sub_grade_rank(test["sub_grade"])
    metrics = {
        "train_loans": len(train),
        "test_loans": len(test),
        "trees": int(model.best_iteration + 1),
        "model": {
            "roc_auc": roc_auc_score(y, test["pd"]),
            "ks_statistic": ks_statistic(y, test["pd"].to_numpy()),
            "brier_score": brier_score_loss(y, test["pd"]),
            "mean_predicted_pd": test["pd"].mean(),
            "observed_default_rate": y.mean(),
        },
        "lending_club_sub_grade": {
            "roc_auc": roc_auc_score(y, grade_rank),
            "ks_statistic": ks_statistic(y, grade_rank.to_numpy()),
        },
        "lending_club_interest_rate": {
            "roc_auc": roc_auc_score(y, test["int_rate"]),
        },
    }
    for block in ("model", "lending_club_sub_grade"):
        metrics[block]["gini"] = 2 * metrics[block]["roc_auc"] - 1

    # Ablation: does adding state unemployment (joined from FRED) help?
    with_macro = MODEL_FEATURES + MACRO_FEATURES
    macro_model = fit_model(train, with_macro)
    macro_pd = macro_model.predict_proba(to_model_frame(test, with_macro))[:, 1]
    metrics["ablation_with_state_unemployment"] = {
        "roc_auc": roc_auc_score(y, macro_pd),
        "mean_predicted_pd": macro_pd.mean(),
    }

    # --- Business layer --------------------------------------------------
    costs = business.DEFAULT_COSTS
    test["economic_profit"] = business.economic_profit(test, costs)
    assumptions = business.estimate_loss_assumptions(train)
    test["expected_loss"] = business.expected_loss(test["pd"], test["funded_amnt"], assumptions)
    test["expected_profit"] = business.expected_profit(
        test["pd"], test["funded_amnt"], test["installment"], assumptions
    )
    test["realized_loss"] = np.where(
        test[TARGET] == 1, test["exposure_at_default"] - test["net_recovery"], 0.0
    )
    metrics["loss_assumptions"] = assumptions.to_dict()
    metrics["cost_assumptions"] = costs.to_dict()

    cutoffs = business.cutoff_table(
        test["pd"], test[TARGET], test["economic_profit"], test["funded_amnt"]
    )
    cutoffs.to_csv(os.path.join(REPORT_DIR, "cutoff_table.csv"), index=False)

    grades_ac = test["grade"].isin(["A", "B", "C"]).to_numpy()
    same_volume = test["pd"].rank(method="first").to_numpy() <= grades_ac.sum()
    positive_ev = (test["expected_profit"] > 0).to_numpy()
    strategies = pd.DataFrame([
        business.summarize_strategy("Approve everything", np.ones(len(test), bool), test),
        business.summarize_strategy("LC grades A-C only", grades_ac, test),
        business.summarize_strategy("Model, same volume as A-C", same_volume, test),
        business.summarize_strategy("Model, expected profit > 0", positive_ev, test),
    ])
    best = cutoffs.loc[cutoffs["profit"].idxmax()]
    metrics["profit_maximizing_approval_rate"] = best["approval_rate"]
    metrics["profit_maximizing_pd_cutoff"] = best["pd_cutoff"]
    strategies.to_csv(os.path.join(REPORT_DIR, "strategies.csv"), index=False)

    # Does expected loss / profit line up with what actually happened?
    test["pd_decile"] = pd.qcut(test["pd"], 10, labels=False) + 1
    el_check = test.groupby("pd_decile").agg(
        loans=("pd", "size"),
        funded=("funded_amnt", "sum"),
        mean_pd=("pd", "mean"),
        default_rate=(TARGET, "mean"),
        expected_loss=("expected_loss", "sum"),
        realized_loss=("realized_loss", "sum"),
        expected_profit=("expected_profit", "sum"),
        realized_profit=("economic_profit", "sum"),
    ).reset_index()
    el_check.to_csv(os.path.join(REPORT_DIR, "expected_vs_realized_loss.csv"), index=False)
    metrics["portfolio_expected_loss"] = test["expected_loss"].sum()
    metrics["portfolio_realized_loss"] = test["realized_loss"].sum()
    metrics["portfolio_expected_profit"] = test["expected_profit"].sum()
    metrics["portfolio_realized_economic_profit"] = test["economic_profit"].sum()

    # --- Macro stress test ---------------------------------------------
    quarterly = macro.load_quarterly()
    sat = macro.fit_satellite(quarterly)
    metrics["macro_satellite"] = sat
    book = test[positive_ev]
    rows = []
    base_ur = sat["latest_unemployment"]
    for name, path in macro.scenarios(base_ur).items():
        shift = macro.lifetime_logit_shift(sat, path, base_ur)
        pd_s = macro.stress_pd(book["pd"], shift)
        el = business.expected_loss(pd_s, book["funded_amnt"], assumptions).sum()
        ep = business.expected_profit(
            pd_s, book["funded_amnt"], book["installment"], assumptions
        ).sum()
        rows.append({
            "scenario": name,
            "peak_unemployment": round(float(path.max()), 1),
            "pd_logit_shift": shift,
            "mean_pd": pd_s.mean(),
            "expected_loss": el,
            "expected_profit": ep,
            "realized_profit_2015": book["economic_profit"].sum() if name == "Baseline" else None,
            "loss_rate": el / book["funded_amnt"].sum(),
        })
    stress = pd.DataFrame(rows)
    stress.to_csv(os.path.join(REPORT_DIR, "stress_scenarios.csv"), index=False)

    # --- Drift monitoring ------------------------------------------------
    # Reference is the most recent training vintage (2014). Using all of
    # 2007-2014 would flag the bureau fields Lending Club only started
    # reporting in 2012 as "drift" purely because they are missing earlier.
    reference = train[train["issue_year"] == 2014].copy()
    drift["pd"] = model.predict_proba(to_model_frame(drift))[:, 1]
    reference["pd"] = model.predict_proba(to_model_frame(reference))[:, 1]
    psi = monitoring.psi_report(
        to_model_frame(reference).assign(pd=reference["pd"]),
        to_model_frame(drift).assign(pd=drift["pd"]),
        ["pd"] + MODEL_FEATURES,
    )
    psi.to_csv(os.path.join(REPORT_DIR, "psi.csv"), index=False)
    metrics["score_psi_2016_2018_vs_train"] = float(psi.loc[psi["variable"] == "pd", "psi"].iloc[0])

    # --- Save --------------------------------------------------------------
    joblib.dump(
        {"model": model, "features": MODEL_FEATURES, "loss_assumptions": assumptions,
         "cost_assumptions": costs, "macro_satellite": sat},
        MODEL_PATH,
    )
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump(_round(metrics), f, indent=2)
    print(json.dumps(_round(metrics), indent=2))
    print(strategies.round(4).to_string(index=False))
    print(stress.round(4).to_string(index=False))
    print(psi.head(8).to_string(index=False))

    _plot_model_vs_grade(y, test["pd"], grade_rank, test["int_rate"])
    _plot_cutoffs(cutoffs, strategies)
    _plot_satellite(quarterly, sat)
    _plot_shap(model, test)
    print(f"\nSaved {MODEL_PATH}, tables in {REPORT_DIR}/, figures in {FIG_DIR}/")


def _round(obj):
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _plot_model_vs_grade(y, model_pd, grade_rank, int_rate):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for label, s in [("This model", model_pd), ("LC sub-grade", grade_rank),
                     ("LC interest rate", int_rate)]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("2015 loans: model vs. Lending Club's grade")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lc_model_vs_grade.png"), dpi=150)
    plt.close(fig)


def _plot_cutoffs(cutoffs, strategies):
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(cutoffs["approval_rate"] * 100, cutoffs["profit"] / 1e6, marker="o",
             color="tab:green", label="Realized economic profit ($M)")
    ax1.set_xlabel("Approval rate (%), lowest predicted PD approved first")
    ax1.set_ylabel("Economic profit after costs ($M)")
    ax2 = ax1.twinx()
    ax2.plot(cutoffs["approval_rate"] * 100, cutoffs["bad_rate"] * 100, marker="s",
             color="tab:red", label="Bad rate (%)")
    ax2.set_ylabel("Bad rate among approved (%)")
    best = cutoffs.loc[cutoffs["profit"].idxmax()]
    ax1.axvline(best["approval_rate"] * 100, color="grey", ls="--", lw=1)
    ax1.annotate(f"max profit at {best['approval_rate']:.0%}",
                 (best["approval_rate"] * 100, best["profit"] / 1e6),
                 textcoords="offset points", xytext=(8, 8), fontsize=9)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower center", fontsize=9)
    ax1.set_title("Approval rate vs. profit and bad rate (2015 loans)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lc_cutoff_curve.png"), dpi=150)
    plt.close(fig)


def _plot_satellite(q, sat):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(q.index, q["chargeoff_rate"], label="Card charge-off rate (%)")
    ax.plot(q.index, q["unemployment"], label="Unemployment rate (%)")
    fitted = pd.Series(100 * macro.fitted_chargeoff(sat, q["unemployment"]), index=q.index)
    covid = (q.index >= "2020-01-01") & (q.index <= "2021-12-31")
    fitted[covid] = np.nan
    ax.plot(q.index, fitted, ls="--", color="grey",
            label=f"Fitted from unemployment (R²={sat['r_squared']:.2f})")
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="grey", alpha=0.15)
    ax.text(pd.Timestamp("2020-03-01"), 12.3, "2020-21\nexcluded", fontsize=8, color="dimgrey")
    ax.set_ylim(0, 14)
    ax.set_ylabel("%")
    ax.set_title("Credit card losses track unemployment (FRED)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lc_macro_satellite.png"), dpi=150)
    plt.close(fig)


def _plot_shap(model, test):
    import shap

    sample = to_model_frame(test.sample(2000, random_state=RANDOM_STATE))
    shap_values = shap.TreeExplainer(model)(sample)
    fig = plt.figure(figsize=(7, 5.5))
    shap.summary_plot(shap_values, sample, show=False, max_display=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "lc_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
