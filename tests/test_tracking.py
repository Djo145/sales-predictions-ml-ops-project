from retail_forecast.tracking import qualification_decision


def test_qualifies_model_that_beats_baseline():
    result = qualification_decision(
        {"mae": 2.0, "rmse": 3.0},
        baseline_mae=2.5,
    )
    assert result["qualification"] == "candidate"


def test_rejects_model_that_does_not_beat_baseline():
    result = qualification_decision(
        {"mae": 3.0, "rmse": 4.0},
        baseline_mae=2.5,
    )
    assert result["qualification"] == "rejected"


def test_rejects_non_finite_metric():
    result = qualification_decision(
        {"mae": float("nan"), "rmse": 4.0},
        baseline_mae=2.5,
    )
    assert result["qualification"] == "rejected"