"""
Portfolio view on 2015 Lending Club loans: pick an approval cutoff, compare
strategies, run a recession scenario, and check drift.

Reads the summary tables written by `python -m src.lc_train`.
"""
import json

import numpy as np
import pandas as pd
import streamlit as st

from src import macro

REPORT_DIR = "reports/lending_club"

st.set_page_config(page_title="Lending Club Portfolio", page_icon="\U0001F4C8", layout="wide")


@st.cache_data
def load():
    tables = {
        name: pd.read_csv(f"{REPORT_DIR}/{name}.csv")
        for name in ["cutoff_table", "strategies", "stress_scenarios", "psi",
                     "expected_vs_realized_loss"]
    }
    with open(f"{REPORT_DIR}/metrics.json") as f:
        return tables, json.load(f)


t, m = load()
money = "${:,.1f}M".format

st.title("\U0001F4C8 Lending Club Portfolio Strategy")
st.caption(
    f"PD model trained on {m['train_loans']:,} 36-month loans issued 2007-2014 and tested "
    f"on {m['test_loans']:,} loans issued in 2015. Profit is realized, after a "
    f"{m['cost_assumptions']['servicing_fee']:.0%} servicing fee and a "
    f"{m['cost_assumptions']['funding_cost_annual']:.0%}/yr funding cost."
)

c1, c2, c3 = st.columns(3)
c1.metric("Model ROC-AUC (2015)", f"{m['model']['roc_auc']:.3f}")
c2.metric("Lending Club sub-grade AUC", f"{m['lending_club_sub_grade']['roc_auc']:.3f}")
c3.metric("Score PSI, 2016-18 vs. 2014", f"{m['score_psi_2016_2018_vs_train']:.3f}")

st.header("Choose an approval cutoff")
cut = t["cutoff_table"]
rate = st.select_slider(
    "Approve the lowest-risk share of applicants",
    options=list(cut["approval_rate"]),
    value=m["profit_maximizing_approval_rate"],
    format_func=lambda r: f"{r:.0%}",
)
row = cut[cut["approval_rate"] == rate].iloc[0]
d1, d2, d3, d4 = st.columns(4)
d1.metric("PD cutoff", f"{row['pd_cutoff']:.1%}")
d2.metric("Bad rate among approved", f"{row['bad_rate']:.1%}")
d3.metric("Economic profit", money(row["profit"] / 1e6))
d4.metric("Return on funded", f"{row['return_on_funded']:.2%}")
st.line_chart(
    cut.assign(**{"Approval rate (%)": cut["approval_rate"] * 100,
                  "Profit ($M)": cut["profit"] / 1e6})
    .set_index("Approval rate (%)")[["Profit ($M)"]]
)

st.header("Strategy comparison")
strat = t["strategies"].copy()
strat["approval_rate"] = strat["approval_rate"].map("{:.1%}".format)
strat["bad_rate"] = strat["bad_rate"].map("{:.1%}".format)
strat["funded"] = (strat["funded"] / 1e6).map(money)
strat["profit"] = (strat["profit"] / 1e6).map(money)
strat["return_on_funded"] = strat["return_on_funded"].map("{:.2%}".format)
st.dataframe(strat, hide_index=True, use_container_width=True)

st.header("Recession scenario")
sat = m["macro_satellite"]
base = sat["latest_unemployment"]
peak = st.slider("Peak unemployment rate (%)", base, 15.0, 10.0, 0.1)
path = macro.unemployment_path(base, peak, 6, 2, base + (peak - base) * 0.6)
shift = macro.lifetime_logit_shift(sat, path, base)
deciles = t["expected_vs_realized_loss"]
loss = m["loss_assumptions"]
loss_factor = loss["lgd"] * loss["ead_ratio"]
el_base = (deciles["mean_pd"] * loss_factor * deciles["funded"]).sum()
el_stress = (macro.stress_pd(deciles["mean_pd"], shift) * loss_factor * deciles["funded"]).sum()
s1, s2, s3 = st.columns(3)
s1.metric("Odds-of-default multiplier", f"{np.exp(shift):.2f}x")
s2.metric("Expected loss, whole 2015 book", money(el_stress / 1e6),
          delta=money((el_stress - el_base) / 1e6), delta_color="inverse")
s3.metric("Loss rate", f"{el_stress / deciles['funded'].sum():.2%}")
st.caption(
    f"Satellite model: logit(card charge-off rate) on unemployment level and 12-month change, "
    f"fit on FRED data 1985-2019 (R² = {sat['r_squared']:.2f}). Assumes Lending Club "
    "borrowers respond to the cycle like bank card borrowers."
)
st.dataframe(t["stress_scenarios"].round(4), hide_index=True, use_container_width=True)

st.header("Drift monitoring (PSI, 2016-2018 applications vs. 2014)")
st.dataframe(t["psi"], hide_index=True, use_container_width=True)
