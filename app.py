"""
Credit Risk Scoring Demo
-------------------------
Interactive Streamlit app: enter an applicant profile, get a predicted
probability of default, a 300-850 scaled risk score, a risk tier, and the
top factors driving that specific prediction (via SHAP).

Run with: streamlit run app.py
"""
import numpy as np
import streamlit as st

from src.predict import score_applicant
from src.scorecard import RISK_TIERS

st.set_page_config(page_title="Credit Risk Scoring Demo", page_icon="\U0001F4B3", layout="wide")

st.title("\U0001F4B3 Credit Risk Scoring Demo")
st.caption(
    "Trained on the *Give Me Some Credit* dataset (~150k consumer credit records) "
    "with an XGBoost classifier. Enter an applicant profile to get a probability of "
    "default, a scaled credit score, and the factors driving the decision."
)

with st.sidebar:
    st.header("Applicant profile")
    age = st.slider("Age", 18, 100, 38)
    monthly_income = st.number_input("Monthly income ($)", 0, 50000, 4500, step=100)
    dependents = st.slider("Number of dependents", 0, 10, 0)
    revolving_util = st.slider(
        "Revolving credit utilization (balance / limit)", 0.0, 1.5, 0.30, step=0.01,
        help="Total balance on credit cards / personal lines, divided by total credit limits.",
    )
    debt_ratio = st.slider(
        "Debt ratio (monthly debt payments / income)", 0.0, 2.0, 0.35, step=0.01,
    )
    open_credit_lines = st.slider("Open credit lines & loans", 0, 30, 8)
    real_estate_loans = st.slider("Real-estate loans / lines", 0, 10, 1)
    st.divider()
    st.caption("Delinquency history (past 2 years)")
    late_30_59 = st.slider("Times 30-59 days past due", 0, 10, 0)
    late_60_89 = st.slider("Times 60-89 days past due", 0, 10, 0)
    late_90 = st.slider("Times 90+ days late", 0, 10, 0)

applicant = {
    "RevolvingUtilizationOfUnsecuredLines": revolving_util,
    "age": age,
    "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
    "DebtRatio": debt_ratio,
    "MonthlyIncome": monthly_income,
    "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
    "NumberOfTimes90DaysLate": late_90,
    "NumberRealEstateLoansOrLines": real_estate_loans,
    "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
    "NumberOfDependents": dependents,
}

result = score_applicant(applicant)
pd_default = result["probability_of_default"]
score = result["score"]
tier = result["tier"]

col1, col2, col3 = st.columns(3)
col1.metric("Probability of default (2yr)", f"{pd_default:.1%}")
col2.metric(
    "Scaled credit score", f"{score:.0f}",
    help="Scaled to 300-850 with points-to-double-the-odds scorecard scaling.",
)
col3.metric("Risk tier", tier)

tier_colors = {
    "Excellent": "#1a9850", "Good": "#66bd63", "Fair": "#fee08b",
    "Poor": "#fc8d59", "Very Poor": "#d73027",
}
st.markdown(
    f"<div style='padding:0.75rem;border-radius:8px;background:{tier_colors.get(tier, '#ddd')}22;"
    f"border:1px solid {tier_colors.get(tier, '#ddd')};'>"
    f"<b>Score range reference:</b> " +
    " &nbsp;|&nbsp; ".join(f"{lo}-{hi}: {label}" for lo, hi, label in RISK_TIERS) +
    "</div>",
    unsafe_allow_html=True,
)

st.subheader("What's driving this prediction?")
if result["reasons"]:
    st.markdown("**Top reasons (adverse action style):** " + "; ".join(result["reasons"]))
contribs = (
    result["contributions"].rename("shap_value").rename_axis("feature").reset_index()
    .sort_values("shap_value", key=abs, ascending=False)
    .head(8)
)
contribs["direction"] = np.where(contribs["shap_value"] > 0, "Increases risk", "Decreases risk")
st.bar_chart(contribs.set_index("feature")["shap_value"])
st.dataframe(
    contribs[["feature", "shap_value", "direction"]], use_container_width=True, hide_index=True
)

with st.expander("About this model"):
    st.markdown(
        """
        - **Data:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
          (Kaggle), 150,000 records, 6.7% default rate.
        - **Model:** XGBoost, trained without class reweighting so the PDs are
          calibrated (mean predicted 6.7% vs. 6.7% observed on the test set).
        - **Test-set performance:** ROC-AUC 0.87, KS 0.58, Gini 0.74
          (see `reports/figures/` and `models/metrics.json`).
        - **Score scaling:** points-to-double-the-odds, 720 at 50:1 odds, PDO 40.
        - **Reasons:** the top SHAP factors that raised risk, worded the way an
          adverse action notice would list them.
        - Built on public benchmark data for a portfolio project, not a
          production underwriting system.
        """
    )
