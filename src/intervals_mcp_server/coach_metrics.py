"""
Deterministic training metrics for the coach report.

This module is pure: no network access, no clock, no logging. The MCP tool in
``tools/coach_report.py`` fetches raw Intervals.icu data and passes it in together
with an explicit end date. Every number an LLM coach needs is computed here so the
model only has to interpret, not calculate.

All thresholds and mappings live in :class:`CoachConfig`.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

SCHEMA_VERSION = "1.0"

Activity = dict[str, Any]
WellnessRecord = dict[str, Any]

RAD = "Rad"
LAUF = "Lauf"
SCHWIMMEN = "Schwimmen"
KRAFT = "Kraft"
SONSTIGE = "Sonstige"

# Intervals.icu activity type -> sport family used throughout the report.
SPORT_FAMILIES: dict[str, str] = {
    "Ride": RAD,
    "VirtualRide": RAD,
    "GravelRide": RAD,
    "MountainBikeRide": RAD,
    "EBikeRide": RAD,
    "Run": LAUF,
    "TrailRun": LAUF,
    "VirtualRun": LAUF,
    "Swim": SCHWIMMEN,
    "OpenWaterSwim": SCHWIMMEN,
    "WeightTraining": KRAFT,
}


@dataclass(frozen=True)
class CoachConfig:  # pylint: disable=too-many-instance-attributes
    """Central place for every threshold, window and mapping used by the report."""

    # Fixed analysis windows (days, inclusive of the end date).
    acute_days: int = 7
    chronic_days: int = 28
    baseline_days: int = 60

    # Load
    acwr_low_warn: float = 0.8
    acwr_high_warn: float = 1.3
    acwr_low_alarm: float = 0.75
    acwr_high_alarm: float = 1.35
    monotony_warn: float = 2.0
    monotony_alarm: float = 2.5
    primary_sport_min_active_days: int = 3
    deload_ratio: float = 0.8
    week_monotony_min_days: int = 5
    week_monotony_min_active_days: int = 3

    # Thresholds and data quality
    ftp_eftp_max_dev_pct: float = 2.0
    sport_inactive_days: int = 7
    tracked_sports: tuple[str, ...] = (RAD, LAUF, SCHWIMMEN, KRAFT)
    subjective_missing_share: float = 0.8

    # Recovery
    hrv_valid_min: float = 10.0
    hrv_valid_max: float = 250.0
    hrv_band_k: float = 0.5
    hrv_min_points_acute: int = 4
    hrv_min_points_baseline: int = 21
    hrv_alarm_days: int = 3
    rhr_high_delta_bpm: float = 5.0
    rhr_warn_days: int = 2
    rhr_alarm_days: int = 3
    rhr_min_points_acute: int = 4
    rhr_min_points_baseline: int = 21
    sleep_short_hours: float = 7.0
    sleep_recent_nights: int = 3
    sleep_short_min_nights: int = 2

    # Intensity: 7-zone -> 3-zone mapping per zone basis (index 0 = Z1 ... 6 = Z7).
    # Power (Coggan zones): Z4 = 91-105 % FTP spans LT2 and counts as high
    # (Treff et al. 2019 / Section 11). HR (LTHR-based zones): Z4 = 94-99 % LTHR is
    # still below LT2 and counts as moderate.
    power_zone_map: tuple[int, ...] = (1, 1, 2, 3, 3, 3, 3)
    hr_zone_map: tuple[int, ...] = (1, 1, 2, 2, 3, 3, 3)
    pi_z2_substitute: float = 0.01
    pi_z3_min: float = 0.01
    pi_polarized_min: float = 2.0
    # Hard-day ladders: (lowest zone, minimum seconds at or above that zone).
    # Power per Seiler/Foster; HR is coarser and lags, so only sustained work
    # above LT2 counts (Section 11).
    hard_day_power_ladder: tuple[tuple[int, int], ...] = (
        (3, 1800),
        (4, 600),
        (5, 300),
        (6, 120),
        (7, 60),
    )
    hard_day_hr_ladder: tuple[tuple[int, int], ...] = ((4, 600), (5, 300))

    # Top sessions
    top_n: int = 5


DEFAULT_CONFIG = CoachConfig()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _num(value: Any) -> float | None:
    """Return ``value`` as float if it is a finite number, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _r(value: float | None, digits: int = 1) -> float | int | None:
    """Round for output; 0 digits yields an int, None stays None."""
    if value is None or not math.isfinite(value):
        return None
    if digits == 0:
        return int(round(value))
    return round(value, digits)


def parse_day(value: Any) -> date | None:
    """Parse the date part of an ISO date/datetime string."""
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def days_back(end: date, count: int) -> list[date]:
    """Return ``count`` consecutive days ending at ``end`` (oldest first)."""
    return [end - timedelta(days=offset) for offset in range(count - 1, -1, -1)]


def sport_family(activity_type: Any) -> str:
    """Map an Intervals.icu activity type to a report sport family."""
    if not isinstance(activity_type, str):
        return SONSTIGE
    return SPORT_FAMILIES.get(activity_type, SONSTIGE)


def activity_day(activity: Activity) -> date | None:
    """Local calendar day of an activity."""
    return parse_day(activity.get("start_date_local"))


def activity_load(activity: Activity) -> float:
    """Training load of an activity; missing or negative loads count as 0."""
    load = _num(activity.get("icu_training_load"))
    return max(load, 0.0) if load is not None else 0.0


def activities_between(activities: list[Activity], start: date, end: date) -> list[Activity]:
    """Activities whose local day lies within [start, end]."""
    selected = []
    for activity in activities:
        day = activity_day(activity)
        if day is not None and start <= day <= end:
            selected.append(activity)
    return selected


def daily_loads(
    activities: list[Activity], end: date, count: int, family: str | None = None
) -> list[float]:
    """Daily training load for ``count`` days ending at ``end``, zero-filled."""
    per_day: dict[date, float] = defaultdict(float)
    for activity in activities:
        if family is not None and sport_family(activity.get("type")) != family:
            continue
        day = activity_day(activity)
        if day is not None:
            per_day[day] += activity_load(activity)
    return [per_day.get(day, 0.0) for day in days_back(end, count)]


def monotony(daily: list[float]) -> float | None:
    """Foster monotony: mean / sample SD of daily load (0-days included).

    Returns None when fewer than two days are given or the SD is 0 (every day
    identical, including a period without any training), because the ratio is
    undefined there.
    """
    if len(daily) < 2:
        return None
    sd = statistics.stdev(daily)
    if sd == 0:
        return None
    return statistics.mean(daily) / sd


def wellness_by_day(wellness: Any) -> dict[date, WellnessRecord]:
    """Index wellness records (list or date-keyed dict from the API) by day."""
    records: list[tuple[Any, Any]] = []
    if isinstance(wellness, dict):
        records = list(wellness.items())
    elif isinstance(wellness, list):
        records = [(item.get("id") if isinstance(item, dict) else None, item) for item in wellness]
    indexed: dict[date, WellnessRecord] = {}
    for key, record in records:
        if not isinstance(record, dict):
            continue
        day = parse_day(record.get("id")) or parse_day(key)
        if day is not None:
            indexed[day] = record
    return indexed


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_metrics(
    activities: list[Activity], end: date, config: CoachConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Acute/chronic load, ACWR, Foster monotony and strain, primary-sport monotony, deload.

    * ACWR = load of the last 7 days / (load of the last 28 days / 4); None if the
      28-day load is 0.
    * Monotony over the last 7 daily loads including rest days (None if undefined,
      see :func:`monotony`); strain = 7-day load x monotony.
    * Primary sport = sport family with the highest 7-day load. Its own monotony is
      only computed with at least ``primary_sport_min_active_days`` active days.
      With more than one sport family, ``effective_monotony`` uses the primary
      sport value, because a steady cross-training floor (e.g. strength) inflates
      total monotony without adding overuse risk (Section 11 approach).
    * Deload: 7-day load <= ``deload_ratio`` x 28-day weekly average.
    """
    acute = daily_loads(activities, end, config.acute_days)
    chronic = daily_loads(activities, end, config.chronic_days)
    load_acute = sum(acute)
    load_chronic = sum(chronic)
    weeks_chronic = config.chronic_days / 7

    acwr = load_acute / (load_chronic / weeks_chronic) if load_chronic > 0 else None
    total_monotony = monotony(acute)
    monotony_note = None
    if total_monotony is None:
        monotony_note = "no_load" if load_acute == 0 else "no_variation"
    strain = load_acute * total_monotony if total_monotony is not None else None

    acute_start = end - timedelta(days=config.acute_days - 1)
    families = sorted(
        {
            sport_family(a.get("type"))
            for a in activities_between(activities, acute_start, end)
            if activity_load(a) > 0
        }
    )
    family_loads = {
        family: daily_loads(activities, end, config.acute_days, family) for family in families
    }
    primary = None
    primary_monotony = None
    if family_loads:
        primary = max(families, key=lambda family: sum(family_loads[family]))
        active_days = sum(1 for load in family_loads[primary] if load > 0)
        if active_days >= config.primary_sport_min_active_days:
            primary_monotony = monotony(family_loads[primary])
    multi_sport = len(family_loads) > 1
    effective = (
        primary_monotony if multi_sport and primary_monotony is not None else total_monotony
    )

    deload = load_chronic > 0 and load_acute <= config.deload_ratio * load_chronic / weeks_chronic

    return {
        "load_7d": _r(load_acute, 0),
        "load_28d": _r(load_chronic, 0),
        "acwr": _r(acwr, 2),
        "monotony": _r(total_monotony, 2),
        "monotony_note": monotony_note,
        "strain": _r(strain, 0),
        "primary_sport": primary,
        "primary_monotony": _r(primary_monotony, 2),
        "effective_monotony": _r(effective, 2),
        "deload": deload,
    }


def fitness_status(
    wellness: dict[date, WellnessRecord], activities: list[Activity], end: date
) -> dict[str, Any]:
    """CTL, ATL, TSB (= CTL - ATL) and ramp rate at the end date.

    Uses the Intervals.icu values of the end date, or the last earlier day that has
    them (reported as ``as_of``). If the end date's ``ctlLoad`` is higher than the
    load actually completed that day, Intervals.icu has counted planned but not yet
    done workouts. CTL/ATL are then recomputed from the previous day with the
    Intervals.icu exponential model (time constants 42 and 7 days) and only the
    completed load; ramp rate becomes CTL(end) - CTL(end - 7 days).
    """
    record_day = None
    for day in days_back(end, 8)[::-1]:
        record = wellness.get(day, {})
        if _num(record.get("ctl")) is not None and _num(record.get("atl")) is not None:
            record_day = day
            break
    if record_day is None:
        return {"ctl": None, "atl": None, "tsb": None, "ramp": None, "src": None}

    record = wellness[record_day]
    ctl = _num(record.get("ctl"))
    atl = _num(record.get("atl"))
    ramp = _num(record.get("rampRate"))
    source = "api"

    if record_day == end:
        planned_load = _num(record.get("ctlLoad"))
        done_load = daily_loads(activities, end, 1)[0]
        previous = wellness.get(end - timedelta(days=1), {})
        prev_ctl = _num(previous.get("ctl"))
        prev_atl = _num(previous.get("atl"))
        if (
            planned_load is not None
            and planned_load > done_load + 1
            and prev_ctl is not None
            and prev_atl is not None
        ):
            ctl = prev_ctl + (done_load - prev_ctl) * (1 - math.exp(-1 / 42))
            atl = prev_atl + (done_load - prev_atl) * (1 - math.exp(-1 / 7))
            week_ago_ctl = _num(wellness.get(end - timedelta(days=7), {}).get("ctl"))
            ramp = ctl - week_ago_ctl if week_ago_ctl is not None else None
            source = "recomputed_without_planned"

    if ctl is None or atl is None:  # unreachable: the search above requires both
        return {"ctl": None, "atl": None, "tsb": None, "ramp": None, "src": None}
    result: dict[str, Any] = {
        "ctl": _r(ctl, 1),
        "atl": _r(atl, 1),
        "tsb": _r(ctl - atl, 1),
        "ramp": _r(ramp, 1),
        "src": source,
    }
    if record_day != end:
        result["as_of"] = record_day.isoformat()
    return result


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------


def iso_week_label(day: date) -> str:
    """ISO week label such as ``2026-W53`` (ISO year, not calendar year)."""
    iso = day.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def weekly_volume(
    activities: list[Activity], end: date, days: int, config: CoachConfig = DEFAULT_CONFIG
) -> list[dict[str, Any]]:
    """Per ISO week (Mon-Sun) and sport family: hours, load, sessions, km.

    Only days inside the report window count, so the first/last week can be partial
    (``days`` < 7). Also returns rest days (daily load 0), hard days (see
    :func:`is_hard_day`) and the week's Foster monotony, which is only reported for
    weeks with enough days and active days (short weeks produce meaningless values).
    """
    window = days_back(end, days)
    by_day: dict[date, list[Activity]] = defaultdict(list)
    for activity in activities_between(activities, window[0], end):
        day = activity_day(activity)
        if day is not None:
            by_day[day].append(activity)

    weeks: dict[str, list[date]] = {}
    for day in window:
        weeks.setdefault(iso_week_label(day), []).append(day)

    rows = []
    for label, week_days in weeks.items():
        sports: dict[str, dict[str, float]] = defaultdict(
            lambda: {"secs": 0.0, "load": 0.0, "n": 0.0, "meters": 0.0}
        )
        daily = []
        hard_days = 0
        for day in week_days:
            if is_hard_day(by_day.get(day, []), config):
                hard_days += 1
            day_load = 0.0
            for activity in by_day.get(day, []):
                totals = sports[sport_family(activity.get("type"))]
                totals["secs"] += _num(activity.get("moving_time")) or 0.0
                totals["load"] += activity_load(activity)
                totals["n"] += 1
                totals["meters"] += _num(activity.get("distance")) or 0.0
                day_load += activity_load(activity)
            daily.append(day_load)

        active_days = sum(1 for load in daily if load > 0)
        week_monotony = None
        if (
            len(daily) >= config.week_monotony_min_days
            and active_days >= config.week_monotony_min_active_days
        ):
            week_monotony = monotony(daily)

        sport_rows: dict[str, dict[str, Any]] = {}
        for family in sorted(sports):
            totals = sports[family]
            row: dict[str, Any] = {
                "h": _r(totals["secs"] / 3600, 1),
                "load": _r(totals["load"], 0),
                "n": int(totals["n"]),
            }
            if totals["meters"] > 0:
                row["km"] = _r(totals["meters"] / 1000, 1)
            sport_rows[family] = row

        rows.append(
            {
                "week": label,
                "days": len(week_days),
                "sports": sport_rows,
                "total": {
                    "h": _r(sum(t["secs"] for t in sports.values()) / 3600, 1),
                    "load": _r(sum(daily), 0),
                    "n": int(sum(t["n"] for t in sports.values())),
                },
                "rest_days": sum(1 for load in daily if load == 0),
                "hard_days": hard_days,
                "monotony": _r(week_monotony, 2),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def _valid_hrv(record: WellnessRecord, config: CoachConfig) -> float | None:
    """rMSSD if it lies in the plausible range, else None (sensor errors)."""
    value = _num(record.get("hrv"))
    if value is None or not config.hrv_valid_min <= value <= config.hrv_valid_max:
        return None
    return value


def hrv_status(
    wellness: dict[date, WellnessRecord], end: date, config: CoachConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """HRV trend: 7-day mean of ln(rMSSD) vs. the baseline normal range.

    Baseline = all valid values in the ``baseline_days`` ending at ``end``; normal
    range = baseline mean +/- ``hrv_band_k`` x baseline SD. Status is
    ``below``/``within``/``above``, or ``insufficient_data`` when the 7-day window
    or the baseline has too few values. ``days_below`` counts consecutive days
    (backwards from ``end``) whose ln(rMSSD) is below the range; a day without a
    value ends the streak.
    """
    baseline_window = days_back(end, config.baseline_days)
    raw = {
        day: value
        for day in baseline_window
        if (value := _valid_hrv(wellness.get(day, {}), config)) is not None
    }
    ln_values = {day: math.log(value) for day, value in raw.items()}
    acute_window = baseline_window[-config.acute_days :]
    acute_ln = [ln_values[day] for day in acute_window if day in ln_values]
    baseline_ln = list(ln_values.values())

    counts = {"n_7d": len(acute_ln), "n_base": len(baseline_ln)}
    if (
        len(acute_ln) < config.hrv_min_points_acute
        or len(baseline_ln) < max(config.hrv_min_points_baseline, 2)
    ):
        return {"status": "insufficient_data", **counts}

    acute_mean = statistics.mean(acute_ln)
    baseline_mean = statistics.mean(baseline_ln)
    baseline_sd = statistics.stdev(baseline_ln)
    lower = baseline_mean - config.hrv_band_k * baseline_sd
    upper = baseline_mean + config.hrv_band_k * baseline_sd
    if acute_mean < lower:
        status = "below"
    elif acute_mean > upper:
        status = "above"
    else:
        status = "within"

    days_below = 0
    for day in reversed(baseline_window):
        if day in ln_values and ln_values[day] < lower:
            days_below += 1
        else:
            break

    return {
        "status": status,
        "ln_7d": _r(acute_mean, 2),
        "ln_base": _r(baseline_mean, 2),
        "band": [_r(lower, 2), _r(upper, 2)],
        "rmssd_7d": _r(statistics.mean(raw[day] for day in acute_window if day in raw), 0),
        "days_below": days_below,
        **counts,
    }


def rhr_status(
    wellness: dict[date, WellnessRecord], end: date, config: CoachConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Resting HR: 7-day mean vs. baseline mean (difference in bpm).

    ``days_high`` counts consecutive days (backwards from ``end``) with a resting
    HR at least ``rhr_high_delta_bpm`` above the baseline mean.
    """
    baseline_window = days_back(end, config.baseline_days)
    values = {}
    for day in baseline_window:
        value = _num(wellness.get(day, {}).get("restingHR"))
        if value is not None and value > 0:
            values[day] = value
    acute = [values[day] for day in baseline_window[-config.acute_days :] if day in values]
    counts = {"n_7d": len(acute), "n_base": len(values)}
    if len(acute) < config.rhr_min_points_acute or len(values) < config.rhr_min_points_baseline:
        return {"status": "insufficient_data", **counts}

    acute_mean = statistics.mean(acute)
    baseline_mean = statistics.mean(values.values())
    days_high = 0
    for day in reversed(baseline_window):
        if day in values and values[day] >= baseline_mean + config.rhr_high_delta_bpm:
            days_high += 1
        else:
            break
    return {
        "avg_7d": _r(acute_mean, 1),
        "avg_base": _r(baseline_mean, 1),
        "delta": _r(acute_mean - baseline_mean, 1),
        "days_high": days_high,
        **counts,
    }


def sleep_summary(
    wellness: dict[date, WellnessRecord],
    end: date,
    days: int,
    config: CoachConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Sleep over the report window: average, 7-day average, short nights, last nights.

    ``recent_h`` lists the last ``sleep_recent_nights`` nights oldest first, with
    None for nights without data.
    """
    hours = {}
    for day in days_back(end, max(days, config.acute_days)):
        secs = _num(wellness.get(day, {}).get("sleepSecs"))
        if secs is not None and secs > 0:
            hours[day] = secs / 3600
    window = [hours[day] for day in days_back(end, days) if day in hours]
    acute = [hours[day] for day in days_back(end, config.acute_days) if day in hours]
    return {
        "avg_h": _r(statistics.mean(window), 1) if window else None,
        "avg_7d_h": _r(statistics.mean(acute), 1) if acute else None,
        "nights_short": sum(1 for value in window if value < config.sleep_short_hours),
        "recent_h": [
            _r(hours[day], 1) if day in hours else None
            for day in days_back(end, config.sleep_recent_nights)
        ],
        "n": len(window),
    }


# ---------------------------------------------------------------------------
# Intensity distribution
# ---------------------------------------------------------------------------

ZONE_COUNT = 7


def _power_zone_secs(activity: Activity) -> list[float] | None:
    """Seconds in power zones Z1..Z7 from ``icu_zone_times``.

    Only the ids Z1..Z7 are used; the extra sweet-spot bucket (id ``SS``) overlaps
    Z3/Z4 and is ignored so time is not counted twice.
    """
    zones = activity.get("icu_zone_times")
    if not isinstance(zones, list):
        return None
    secs = [0.0] * ZONE_COUNT
    for zone in zones:
        if not isinstance(zone, dict) or not isinstance(zone.get("id"), str):
            continue
        zone_id = zone["id"].strip().upper()
        if len(zone_id) == 2 and zone_id[0] == "Z" and zone_id[1] in "1234567":
            secs[int(zone_id[1]) - 1] += max(_num(zone.get("secs")) or 0.0, 0.0)
    return secs if sum(secs) > 0 else None


def _hr_zone_secs(activity: Activity) -> list[float] | None:
    """Seconds in HR zones Z1..Z7 from the ``icu_hr_zone_times`` array (index = zone)."""
    zones = activity.get("icu_hr_zone_times")
    if not isinstance(zones, list):
        return None
    secs = [0.0] * ZONE_COUNT
    for index, value in enumerate(zones[:ZONE_COUNT]):
        secs[index] = max(_num(value) or 0.0, 0.0)
    return secs if sum(secs) > 0 else None


def activity_zones(activity: Activity) -> tuple[list[float] | None, str | None]:
    """Zone seconds (Z1..Z7) and their basis (``power`` or ``hr``) for one activity.

    Cycling uses power zones with HR as fallback; every other sport uses HR zones
    with power as fallback. Returns (None, None) without any zone data.
    """
    power = _power_zone_secs(activity)
    heart_rate = _hr_zone_secs(activity)
    candidates = [(power, "power"), (heart_rate, "hr")]
    if sport_family(activity.get("type")) != RAD:
        candidates.reverse()
    for secs, basis in candidates:
        if secs is not None:
            return secs, basis
    return None, None


def three_zone_seconds(
    secs: list[float], basis: str, config: CoachConfig = DEFAULT_CONFIG
) -> list[float]:
    """Collapse Z1..Z7 seconds into the 3-zone model with the basis-specific mapping."""
    mapping = config.power_zone_map if basis == "power" else config.hr_zone_map
    result = [0.0, 0.0, 0.0]
    for index, value in enumerate(secs):
        result[mapping[index] - 1] += value
    return result


def polarization_index(
    z1: float, z2: float, z3: float, config: CoachConfig = DEFAULT_CONFIG
) -> tuple[float | None, str | None]:
    """Polarization index after Treff et al. (2019), Front Physiol 10:707.

    PI = log10((Z1 / Z2) x Z3 x 100) with Z1..Z3 as fractions of total time.
    Edge cases (returned as a note):

    * no zone time at all -> (None, ``no_zone_data``)
    * Z3 < ``pi_z3_min`` (1 %) -> (None, ``z3_zero``): the logarithm is undefined
      for Z3 = 0 and such a distribution is by definition not polarized
    * Z1 = 0 -> (None, ``z1_zero``) for the same reason
    * Z2 = 0 -> Z2 is replaced by ``pi_z2_substitute`` (0.01) as proposed by Treff
      et al., note ``z2_substituted``

    The index is returned for any distribution where it is defined; whether the
    structure is actually polarized is decided in :func:`classify_tid`.
    """
    if z1 + z2 + z3 <= 0:
        return None, "no_zone_data"
    if z3 < config.pi_z3_min:
        return None, "z3_zero"
    if z1 <= 0:
        return None, "z1_zero"
    note = None
    if z2 <= 0:
        z2 = config.pi_z2_substitute
        note = "z2_substituted"
    return math.log10(z1 / z2 * z3 * 100), note


def classify_tid(
    z1: float, z2: float, z3: float, pi: float | None, config: CoachConfig = DEFAULT_CONFIG
) -> str:
    """Training intensity distribution class (Section 11 priority order).

    1. Base: Z3 below 1 % and Z1 largest
    2. Polarized: Z1 > Z3 > Z2 and PI > 2.0
    3. Pyramidal: Z1 > Z2 > Z3
    4. Threshold: Z2 largest
    5. HIT: Z3 largest
    Anything else (e.g. Z1 > Z3 > Z2 with PI <= 2.0) is reported as Pyramidal.
    """
    if z3 < config.pi_z3_min and z1 >= z2:
        return "Base"
    if z1 > z3 > z2 and pi is not None and pi > config.pi_polarized_min:
        return "Polarized"
    if z1 > z2 > z3:
        return "Pyramidal"
    if z2 >= z1 and z2 >= z3:
        return "Threshold"
    if z3 >= z1 and z3 >= z2:
        return "HIT"
    return "Pyramidal"


def intensity_distribution(
    activities: list[Activity], config: CoachConfig = DEFAULT_CONFIG, family: str | None = None
) -> dict[str, Any]:
    """3-zone distribution, polarization index and TID class over the given activities.

    Power and HR zone times are mapped separately (each with its own mapping) and
    then summed; ``basis_pct`` shows how much of the time came from each basis.
    """
    totals = [0.0, 0.0, 0.0]
    by_basis = {"power": 0.0, "hr": 0.0}
    for activity in activities:
        if family is not None and sport_family(activity.get("type")) != family:
            continue
        secs, basis = activity_zones(activity)
        if secs is None or basis is None:
            continue
        collapsed = three_zone_seconds(secs, basis, config)
        for index in range(3):
            totals[index] += collapsed[index]
        by_basis[basis] += sum(collapsed)

    total = sum(totals)
    if total <= 0:
        return {"pct": None, "h": 0, "pi": None, "pi_note": "no_zone_data", "cls": None}
    z1, z2, z3 = (value / total for value in totals)
    pi, note = polarization_index(z1, z2, z3, config)
    result: dict[str, Any] = {
        "pct": [_r(z1 * 100, 1), _r(z2 * 100, 1), _r(z3 * 100, 1)],
        "h": _r(total / 3600, 1),
        "pi": _r(pi, 2),
        "cls": classify_tid(z1, z2, z3, pi, config),
        "basis_pct": {basis: _r(secs / total * 100, 0) for basis, secs in by_basis.items()},
    }
    if note is not None:
        result["pi_note"] = note
    return result


def tid_drift(
    activities: list[Activity], end: date, config: CoachConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Compare the 7-day with the 28-day distribution (all sports).

    ``acute_depolarization``: PI dropped below the polarized threshold in the last
    7 days while the 28-day PI is above it; ``shifting``: the class changed;
    ``consistent``: same class. None when either window lacks zone data.
    """
    acute = intensity_distribution(
        activities_between(activities, end - timedelta(days=config.acute_days - 1), end), config
    )
    chronic = intensity_distribution(
        activities_between(activities, end - timedelta(days=config.chronic_days - 1), end), config
    )
    cls_acute, cls_chronic = acute["cls"], chronic["cls"]
    pi_acute, pi_chronic = acute["pi"], chronic["pi"]
    drift = None
    if cls_acute is not None and cls_chronic is not None:
        if (
            pi_acute is not None
            and pi_chronic is not None
            and pi_acute < config.pi_polarized_min <= pi_chronic
        ):
            drift = "acute_depolarization"
        elif cls_acute != cls_chronic:
            drift = "shifting"
        else:
            drift = "consistent"
    return {
        "cls_7d": cls_acute,
        "cls_28d": cls_chronic,
        "pi_7d": pi_acute,
        "pi_28d": pi_chronic,
        "drift": drift,
    }


def is_hard_day(day_activities: list[Activity], config: CoachConfig = DEFAULT_CONFIG) -> bool | None:
    """Whether a day counts as hard, from the day's summed zone times.

    Power and HR zone times are accumulated separately. A day is hard if any rung
    of the power ladder or the HR ladder is reached (time at or above a zone).
    Returns None when no activity of the day has zone data.
    """
    by_basis: dict[str, list[float]] = {}
    for activity in day_activities:
        secs, basis = activity_zones(activity)
        if secs is None or basis is None:
            continue
        accumulated = by_basis.setdefault(basis, [0.0] * ZONE_COUNT)
        for index, value in enumerate(secs):
            accumulated[index] += value
    if not by_basis:
        return None
    ladders = {"power": config.hard_day_power_ladder, "hr": config.hard_day_hr_ladder}
    for basis, secs in by_basis.items():
        for zone, min_secs in ladders[basis]:
            if sum(secs[zone - 1 :]) >= min_secs:
                return True
    return False
