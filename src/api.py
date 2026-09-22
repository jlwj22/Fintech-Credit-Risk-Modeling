"""
REST API for scoring applicants with the Give Me Some Credit model.

    uvicorn src.api:app --reload
    curl -X POST localhost:8000/score -H 'Content-Type: application/json' \
         -d '{"revolving_utilization": 0.45, "age": 34, "debt_ratio": 0.3,
              "monthly_income": 5200, "open_credit_lines": 7}'

Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import load_bundle, score_applicant

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Probability of default, 300-850 score, risk tier and reason codes.",
    version="1.0.0",
)


class Applicant(BaseModel):
    revolving_utilization: float = Field(
        ..., ge=0, description="Revolving balances / credit limits (0.3 = 30%)"
    )
    age: int = Field(..., ge=18, le=110)
    debt_ratio: float = Field(..., ge=0, description="Monthly debt payments / monthly income")
    monthly_income: float | None = Field(None, ge=0, description="Leave empty if unknown")
    open_credit_lines: int = Field(..., ge=0)
    real_estate_loans: int = Field(0, ge=0)
    dependents: int | None = Field(0, ge=0)
    times_30_59_days_late: int = Field(0, ge=0)
    times_60_89_days_late: int = Field(0, ge=0)
    times_90_days_late: int = Field(0, ge=0)

    def to_raw(self) -> dict:
        return {
            "RevolvingUtilizationOfUnsecuredLines": self.revolving_utilization,
            "age": self.age,
            "NumberOfTime30-59DaysPastDueNotWorse": self.times_30_59_days_late,
            "DebtRatio": self.debt_ratio,
            "MonthlyIncome": self.monthly_income,
            "NumberOfOpenCreditLinesAndLoans": self.open_credit_lines,
            "NumberOfTimes90DaysLate": self.times_90_days_late,
            "NumberRealEstateLoansOrLines": self.real_estate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": self.times_60_89_days_late,
            "NumberOfDependents": self.dependents,
        }


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: int
    tier: str
    reasons: list[str] = Field(description="Top factors that raised this applicant's risk")


@app.get("/health")
def health() -> dict:
    load_bundle()
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant) -> ScoreResponse:
    result = score_applicant(applicant.to_raw())
    return ScoreResponse(
        probability_of_default=round(result["probability_of_default"], 4),
        score=result["score"],
        tier=result["tier"],
        reasons=result["reasons"],
    )
