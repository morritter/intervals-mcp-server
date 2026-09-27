"""
Unit tests for the pure coach_metrics module (no network).
"""

import math
from datetime import date

import pytest

from intervals_mcp_server import coach_metrics as cm
from tests.coach_fixtures import END, activity, days_before, wellness


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sport_family_merges_ride_types():
    assert cm.sport_family("Ride") == cm.RAD
    assert cm.sport_family("VirtualRide") == cm.RAD
    assert cm.sport_family("Run") == cm.LAUF
    assert cm.sport_family("Swim") == cm.SCHWIMMEN
    assert cm.sport_family("WeightTraining") == cm.KRAFT
    assert cm.sport_family("Yoga") == cm.SONSTIGE
    assert cm.sport_family(None) == cm.SONSTIGE


def test_daily_loads_zero_fills_days_without_training():
    acts = [activity(END, load=50), activity(END, load=30, hour=18), activity(days_before(END, 2), load=20)]
    assert cm.daily_loads(acts, END, 4) == [0.0, 20.0, 0.0, 80.0]


def test_daily_loads_treats_missing_load_as_zero():
    acts = [activity(END, load=None), activity(END, load=-5)]
    assert cm.daily_loads(acts, END, 1) == [0.0]


def test_monotony_undefined_cases():
    assert cm.monotony([]) is None
    assert cm.monotony([50.0]) is None
    assert cm.monotony([0.0] * 7) is None  # no training
    assert cm.monotony([60.0] * 7) is None  # SD = 0 -> division by zero avoided


def test_monotony_includes_rest_days():
    daily = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0]
    expected = (400 / 7) / math.sqrt(sum((x - 400 / 7) ** 2 for x in daily) / 6)
    assert cm.monotony(daily) == pytest.approx(expected)


def test_wellness_by_day_accepts_list_and_dict():
    record = wellness(END)
    assert cm.wellness_by_day([record])[END] is record
    assert END in cm.wellness_by_day({END.isoformat(): {"hrv": 70}})
    assert cm.wellness_by_day(None) == {}


def test_iso_week_label_crosses_year_boundary():
    assert cm.iso_week_label(date(2027, 1, 1)) == "2026-W53"
    assert cm.iso_week_label(date(2027, 1, 4)) == "2027-W01"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def test_load_metrics_acwr_monotony_strain():
    acts = [activity(days_before(END, offset), load=100) for offset in range(28) if offset % 2 == 0]
    result = cm.load_metrics(acts, END)
    assert result["load_7d"] == 400
    assert result["load_28d"] == 1400
    assert result["acwr"] == round(400 / (1400 / 4), 2)
    mono = cm.monotony([100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0])
    assert mono is not None
    assert result["monotony"] == round(mono, 2)
    assert result["strain"] == round(400 * mono)
    assert result["monotony_note"] is None


def test_load_metrics_empty_period_avoids_division_by_zero():
    result = cm.load_metrics([], END)
    assert result["acwr"] is None
    assert result["monotony"] is None
    assert result["monotony_note"] == "no_load"
    assert result["strain"] is None
    assert result["primary_sport"] is None
    assert result["deload"] is False


def test_load_metrics_identical_daily_load():
    acts = [activity(days_before(END, offset), load=60) for offset in range(7)]
    result = cm.load_metrics(acts, END)
    assert result["monotony"] is None
    assert result["monotony_note"] == "no_variation"
    assert result["strain"] is None


def test_primary_sport_monotony_removes_cross_training_floor():
    acts = []
    for offset in range(7):
        acts.append(activity(days_before(END, offset), "WeightTraining", load=15, hour=18))
    for offset, load in [(0, 200), (1, 50), (3, 120), (5, 60)]:
        acts.append(activity(days_before(END, offset), "Ride", load=load))
    result = cm.load_metrics(acts, END)
    assert result["primary_sport"] == cm.RAD
    assert result["primary_monotony"] is not None
    assert result["effective_monotony"] == result["primary_monotony"]
    assert result["primary_monotony"] < result["monotony"]


def test_primary_monotony_needs_three_active_days():
    acts = [activity(days_before(END, 0), load=200), activity(days_before(END, 3), load=100)]
    acts += [activity(days_before(END, offset), "Run", load=10) for offset in range(7)]
    result = cm.load_metrics(acts, END)
    assert result["primary_sport"] == cm.RAD
    assert result["primary_monotony"] is None
    assert result["effective_monotony"] == result["monotony"]


def test_deload_detected():
    acts = [activity(days_before(END, offset), load=100) for offset in range(7, 28)]
    acts += [activity(days_before(END, offset), load=40) for offset in range(7)]
    assert cm.load_metrics(acts, END)["deload"] is True


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------


def test_fitness_uses_api_values_and_tsb():
    days = {END: wellness(END, ctl=101.23, atl=93.9, ramp=-4.9)}
    result = cm.fitness_status(days, [], END)
    assert result == {"ctl": 101.2, "atl": 93.9, "tsb": 7.3, "ramp": -4.9, "src": "api"}


def test_fitness_falls_back_to_latest_day():
    yesterday = days_before(END, 1)
    days = {yesterday: wellness(yesterday, ctl=100, atl=90), END: wellness(END, ctl=None, atl=None)}
    result = cm.fitness_status(days, [], END)
    assert result["ctl"] == 100
    assert result["as_of"] == yesterday.isoformat()


def test_fitness_missing_entirely():
    assert cm.fitness_status({}, [], END)["ctl"] is None


def test_fitness_recomputed_when_planned_load_is_counted():
    week_ago = days_before(END, 7)
    yesterday = days_before(END, 1)
    days = {
        week_ago: wellness(week_ago, ctl=95.0),
        yesterday: wellness(yesterday, ctl=100.0, atl=110.0),
        END: wellness(END, ctl=102.0, atl=120.0, ctl_load=150.0),
    }
    result = cm.fitness_status(days, [], END)
    assert result["src"] == "recomputed_without_planned"
    assert result["ctl"] == round(100.0 * math.exp(-1 / 42), 1)
    assert result["atl"] == round(110.0 * math.exp(-1 / 7), 1)
    assert result["ramp"] == round(100.0 * math.exp(-1 / 42) - 95.0, 1)


def test_fitness_not_recomputed_when_load_was_done():
    yesterday = days_before(END, 1)
    days = {
        yesterday: wellness(yesterday, ctl=100.0, atl=110.0),
        END: wellness(END, ctl=102.0, atl=120.0, ctl_load=150.0),
    }
    result = cm.fitness_status(days, [activity(END, load=150)], END)
    assert result["src"] == "api"
    assert result["ctl"] == 102.0


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def test_weekly_volume_groups_iso_weeks_and_merges_rad():
    acts = [
        activity(END, "Ride", load=100, moving=7200, distance=60000),
        activity(END, "VirtualRide", load=50, moving=3600, distance=None, hour=18),
        activity(days_before(END, 1), "Run", load=60, moving=3600, distance=10000),
        activity(days_before(END, 3), "WeightTraining", load=10, moving=1800, distance=None),
    ]
    rows = cm.weekly_volume(acts, END, 14)
    assert [row["week"] for row in rows] == ["2026-W38", "2026-W39"]
    week = rows[1]
    assert week["days"] == 7
    assert week["sports"][cm.RAD] == {"h": 3.0, "load": 150, "n": 2, "km": 60.0}
    assert week["sports"][cm.LAUF] == {"h": 1.0, "load": 60, "n": 1, "km": 10.0}
    assert week["sports"][cm.KRAFT] == {"h": 0.5, "load": 10, "n": 1}
    assert week["total"] == {"h": 4.5, "load": 220, "n": 4}
    assert week["rest_days"] == 4
    assert week["monotony"] == round(cm.monotony([0, 0, 0, 10, 0, 60, 150]) or 0, 2)
    assert rows[0]["sports"] == {}
    assert rows[0]["rest_days"] == 7
    assert rows[0]["monotony"] is None


def test_weekly_volume_partial_week_has_no_monotony():
    acts = [activity(days_before(END, offset), load=50 + offset) for offset in range(9)]
    rows = cm.weekly_volume(acts, END, 9)
    assert rows[0]["days"] == 2
    assert rows[0]["monotony"] is None
    assert rows[1]["monotony"] is not None
