import pandas as pd

from proctor_engine import ExamProctorConfig, aggregate_events, compute_risk_score


def test_risk_score_zero_when_no_flags():
    assert compute_risk_score({"multiple_people": False, "cell_phone_visible": False}) == 0.0


def test_event_aggregation_multiple_people():
    cfg = ExamProctorConfig(multiple_people_min_duration_sec=1.0, event_max_gap_sec=0.2)
    df = pd.DataFrame(
        {
            "time_sec": [0.0, 0.5, 1.0, 1.5, 2.0],
            "risk_score": [0.2, 0.2, 0.2, 0.0, 0.0],
            "flag_multiple_people": [True, True, True, False, False],
        }
    )
    events = aggregate_events(df, cfg)
    assert len(events) == 1
    assert events.iloc[0]["flag"] == "multiple_people"
    assert events.iloc[0]["duration_sec"] == 1.0
