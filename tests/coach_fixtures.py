"""
Synthetic Intervals.icu data for the coach report tests.

Field names mirror the real API (activities, wellness, athlete). Values are made up;
no athlete IDs or API keys appear here.
"""

from datetime import date, timedelta
from typing import Any

END = date(2026, 9, 27)  # a Sunday, so 28 days = exactly 4 ISO weeks


def day_str(day: date, hour: int = 9) -> str:
    """Local start timestamp as returned in ``start_date_local``."""
    return f"{day.isoformat()}T{hour:02d}:00:00"


def power_zones(*secs: float, sweet_spot: float = 0.0) -> list[dict[str, Any]]:
    """``icu_zone_times`` for Z1..Zn plus the overlapping sweet-spot bucket."""
    zones = [{"id": f"Z{index}", "secs": value} for index, value in enumerate(secs, start=1)]
    zones.append({"id": "SS", "secs": sweet_spot})
    return zones


def activity(  # pylint: disable=too-many-arguments
    day: date,
    type_: str = "Ride",
    *,
    load: float | None = 80,
    moving: float = 7200,
    elapsed: float | None = None,
    distance: float | None = 60000,
    vi: float | None = 1.05,
    decoupling: float | None = 3.0,
    temp: float | None = 18.0,
    intensity: float | None = 70.0,
    ef: float | None = 1.8,
    zones: list[dict[str, Any]] | None = None,
    hr_zones: list[float] | None = None,
    rpe: int | None = None,
    feel: int | None = None,
    hour: int = 9,
) -> dict[str, Any]:
    """Build one activity dict with real Intervals.icu field names."""
    return {
        "type": type_,
        "start_date_local": day_str(day, hour),
        "icu_training_load": load,
        "moving_time": moving,
        "elapsed_time": elapsed if elapsed is not None else moving,
        "distance": distance,
        "icu_variability_index": vi,
        "decoupling": decoupling,
        "average_temp": temp,
        "icu_intensity": intensity,
        "icu_efficiency_factor": ef,
        "icu_zone_times": zones,
        "icu_hr_zone_times": hr_zones,
        "icu_rpe": rpe,
        "feel": feel,
    }


def wellness(  # pylint: disable=too-many-arguments
    day: date,
    *,
    hrv: float | None = 80.0,
    rhr: float | None = 40.0,
    sleep_h: float | None = 7.5,
    ctl: float | None = 100.0,
    atl: float | None = 100.0,
    ramp: float | None = 1.0,
    ctl_load: float | None = None,
    eftp: float | None = 352.0,
) -> dict[str, Any]:
    """Build one wellness record keyed by ``id`` = date."""
    return {
        "id": day.isoformat(),
        "hrv": hrv,
        "restingHR": rhr,
        "sleepSecs": sleep_h * 3600 if sleep_h is not None else None,
        "ctl": ctl,
        "atl": atl,
        "rampRate": ramp,
        "ctlLoad": ctl_load,
        "sportInfo": [{"type": "Ride", "eftp": eftp}] if eftp is not None else [],
    }


def athlete(ftp: int = 343, indoor_ftp: int | None = None) -> dict[str, Any]:
    """Athlete record with sport settings like ``GET /athlete/{id}``."""
    return {
        "timezone": "Europe/Berlin",
        "sportSettings": [
            {"types": ["Ride", "VirtualRide"], "ftp": ftp, "indoor_ftp": indoor_ftp, "lthr": 170},
            {"types": ["Run", "VirtualRun", "TrailRun"], "ftp": None, "lthr": 173},
            {"types": ["Swim"], "ftp": None, "lthr": None},
        ],
    }


def days_before(end: date, offset: int) -> date:
    """Day ``offset`` days before ``end``."""
    return end - timedelta(days=offset)


def realistic_scenario(end: date = END) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """~15 h/week multi-sport athlete over 60 days: Rad, Lauf, Schwimmen, Kraft."""
    activities: list[dict[str, Any]] = []
    for offset in range(60):
        day = days_before(end, offset)
        weekday = day.weekday()
        if weekday == 0:
            continue  # Monday rest day
        if weekday in (1, 3):
            activities.append(
                activity(
                    day,
                    "Ride",
                    load=110,
                    moving=7200,
                    elapsed=7600,
                    vi=1.08,
                    decoupling=4.2,
                    intensity=78.0,
                    zones=power_zones(1500, 3000, 900, 1200, 480, 90, 30, sweet_spot=900),
                    hr_zones=[2000, 3200, 1100, 700, 200],
                )
            )
            activities.append(
                activity(day, "WeightTraining", load=15, moving=2400, distance=None, vi=None,
                         decoupling=None, temp=None, intensity=45.0, ef=None,
                         hr_zones=[2000, 400, 0, 0, 0], hour=18)
            )
        elif weekday == 2:
            activities.append(
                activity(day, "Run", load=60, moving=3600, distance=11000, vi=None,
                         decoupling=None, temp=24.0, intensity=75.0, ef=None,
                         hr_zones=[400, 2600, 500, 100, 0, 0, 0])
            )
            activities.append(
                activity(day, "Swim", load=20, moving=2700, distance=2500, vi=None,
                         decoupling=None, temp=None, intensity=52.0, ef=None,
                         hr_zones=[1500, 1000, 200, 0, 0], hour=17)
            )
        elif weekday == 4:
            activities.append(
                activity(day, "VirtualRide", load=70, moving=3600, distance=None, vi=1.03,
                         decoupling=2.1, temp=None, intensity=72.0, ef=1.9,
                         zones=power_zones(300, 2400, 600, 300, 0, 0, 0))
            )
        elif weekday == 5:
            activities.append(
                activity(day, "Ride", load=260, moving=18000, elapsed=19800, distance=150000,
                         vi=1.12, decoupling=5.5, temp=21.0, intensity=66.0,
                         zones=power_zones(4000, 10000, 2500, 1000, 300, 150, 50))
            )
        else:  # Sunday
            activities.append(
                activity(day, "Run", load=85, moving=5400, distance=16000, vi=None,
                         decoupling=None, temp=15.0, intensity=70.0, ef=None,
                         hr_zones=[800, 4200, 300, 100, 0, 0, 0])
            )
    records = [
        wellness(
            days_before(end, offset),
            hrv=78.0 + (offset % 5),
            rhr=40.0 + (offset % 3),
            sleep_h=6.6 if offset in (0, 2) else 7.4,
            ctl=100.0 - offset * 0.05,
            atl=95.0,
            eftp=352.0 - offset * 0.1,
        )
        for offset in range(60)
    ]
    return activities, records, athlete()
