from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

APPLICANT = {
    "revolving_utilization": 0.45,
    "age": 34,
    "debt_ratio": 0.3,
    "monthly_income": 5200,
    "open_credit_lines": 7,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_returns_valid_response():
    body = client.post("/score", json=APPLICANT).json()
    assert 0 < body["probability_of_default"] < 1
    assert 300 <= body["score"] <= 850
    assert body["tier"] in {"Excellent", "Good", "Fair", "Poor", "Very Poor"}
    assert isinstance(body["reasons"], list)


def test_missing_income_is_accepted():
    body = client.post("/score", json={**APPLICANT, "monthly_income": None}).json()
    assert 0 < body["probability_of_default"] < 1


def test_delinquency_raises_risk_and_is_explained():
    clean = client.post("/score", json=APPLICANT).json()
    late = client.post("/score", json={**APPLICANT, "times_90_days_late": 3}).json()
    assert late["probability_of_default"] > clean["probability_of_default"]
    assert "Serious delinquency (90+ days late)" in late["reasons"]


def test_invalid_input_rejected():
    assert client.post("/score", json={**APPLICANT, "age": 12}).status_code == 422
