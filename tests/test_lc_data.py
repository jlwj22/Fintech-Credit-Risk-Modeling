import numpy as np
import pandas as pd

from src.lc_data import MODEL_FEATURES, TARGET, engineer, to_model_frame


def _loans():
    base = {
        "loan_amnt": 10000.0, "funded_amnt": 10000.0, "term": " 36 months",
        "int_rate": 12.0, "installment": 332.0, "grade": "B", "sub_grade": "B3",
        "emp_length": "10+ years", "home_ownership": "MORTGAGE", "annual_inc": 60000.0,
        "verification_status": "Verified", "issue_d": "Mar-2014",
        "purpose": "credit_card", "addr_state": "CA", "dti": 15.0, "delinq_2yrs": 0.0,
        "earliest_cr_line": "Mar-2004", "fico_range_low": 700.0, "fico_range_high": 704.0,
        "inq_last_6mths": 1.0, "mths_since_last_delinq": np.nan,
        "mths_since_last_record": np.nan, "open_acc": 8.0, "pub_rec": 0.0,
        "revol_bal": 5000.0, "revol_util": 40.0, "total_acc": 20.0, "mort_acc": 1.0,
        "pub_rec_bankruptcies": 0.0, "application_type": "Individual",
        "total_pymnt": 11900.0, "total_rec_prncp": 10000.0, "total_rec_int": 1900.0,
        "recoveries": 0.0, "collection_recovery_fee": 0.0,
        "acc_open_past_24mths": 3.0, "bc_util": 45.0, "percent_bc_gt_75": 20.0,
        "num_tl_op_past_12m": 1.0, "mths_since_recent_inq": 4.0, "mo_sin_rcnt_tl": 5.0,
        "avg_cur_bal": 12000.0, "total_rev_hi_lim": 20000.0, "num_actv_rev_tl": 4.0,
        "pct_tl_nvr_dlq": 100.0,
    }
    charged_off = {**base, "loan_status": "Charged Off", "emp_length": "< 1 year",
                   "total_pymnt": 4000.0, "total_rec_prncp": 3000.0,
                   "total_rec_int": 900.0, "recoveries": 500.0,
                   "collection_recovery_fee": 90.0, "home_ownership": "NONE"}
    current = {**base, "loan_status": "Current"}
    return pd.DataFrame([{**base, "loan_status": "Fully Paid"}, charged_off, current])


def test_labels():
    out = engineer(_loans())
    assert out[TARGET].iloc[0] == 0
    assert out[TARGET].iloc[1] == 1
    assert np.isnan(out[TARGET].iloc[2])


def test_engineered_fields():
    out = engineer(_loans())
    assert abs(out["credit_history_years"].iloc[0] - 10.0) < 0.01
    assert out["fico"].iloc[0] == 702.0
    assert out["emp_years"].tolist() == [10.0, 0.0, 10.0]
    assert out["loan_to_income"].iloc[0] == 10000 / 60000
    assert out["home_ownership"].iloc[1] == "OTHER"


def test_realized_economics():
    out = engineer(_loans()).iloc[1]
    assert out["exposure_at_default"] == 7000.0
    assert out["net_recovery"] == 410.0
    assert out["realized_profit"] == 4000.0 - 90.0 - 10000.0


def test_model_frame_has_every_feature_and_fixed_categories():
    X = to_model_frame(engineer(_loans()))
    assert list(X.columns) == MODEL_FEATURES
    assert "RENT" in X["home_ownership"].cat.categories
