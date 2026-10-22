"""Synthetic daily demand for 6 medicines on a hospital ward, 2023-2025.

Demand depends on the weekday, the season, a long-term trend and a regional flu activity index
(observed with a one-day delay, like public surveillance data). Counts follow a negative binomial
distribution, so demand is noisier than a Poisson process, as real hospital demand is.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Medicine:
    name: str
    base: float          # mean daily demand (packs) outside flu season
    flu_effect: float    # extra demand per unit of flu index
    volume: float        # cabinet space per pack (litres)
    holding: float       # EUR per pack left over (return handling, expiry risk)
    shortage: float      # EUR per pack short (emergency delivery from central pharmacy)


MEDICINES = [
    Medicine("Antibiotic IV", 22, 0.9, 0.6, 1.0, 12.0),
    Medicine("Antiviral oral", 6, 3.0, 0.2, 2.0, 18.0),
    Medicine("Paracetamol IV", 30, 0.5, 0.5, 0.5, 4.0),
    Medicine("Insulin", 12, 0.05, 0.3, 1.5, 25.0),
    Medicine("Heparin", 15, 0.1, 0.4, 1.0, 15.0),
    Medicine("Saline 500 ml", 45, 0.3, 0.6, 0.3, 3.0),
]
CABINET_CAPACITY = 110.0  # litres of space for next-day stock; becomes tight during flu season
DOW_FACTOR = np.array([1.08, 1.05, 1.02, 1.0, 0.98, 0.80, 0.78])  # Mon..Sun


def make_data(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.to_numpy()

    # Flu index 0..~4: winter wave with a different peak date and strength each year, plus noise
    flu = np.zeros(n)
    for year in [2022, 2023, 2024, 2025]:
        peak = pd.Timestamp(f"{year + 1}-01-01") + pd.Timedelta(days=int(rng.integers(10, 60)))
        strength = rng.uniform(1.8, 4.0)
        days = (dates - peak).days.to_numpy()
        flu += strength * np.exp(-(days / 28.0) ** 2)
    flu = np.clip(flu + rng.normal(0, 0.15, n), 0, None)

    season = 1 + 0.06 * np.cos((doy - 15) / 365 * 2 * np.pi)
    trend = 1 + 0.04 * np.arange(n) / 365  # +4% per year (ageing population)

    rows = {"date": dates, "flu_index": flu}
    for m in MEDICINES:
        mean = m.base * DOW_FACTOR[dates.dayofweek] * season * trend * (1 + m.flu_effect * flu / 4)
        k = 12.0  # negative binomial dispersion: variance = mean + mean^2 / k
        rows[m.name] = rng.negative_binomial(k, k / (k + mean))
    return pd.DataFrame(rows)


def features(df: pd.DataFrame, med: str) -> pd.DataFrame:
    """Features known the day before: calendar, yesterday's flu index, recent demand."""
    X = pd.DataFrame(index=df.index)
    X["dow"] = df["date"].dt.dayofweek
    for d in range(6):
        X[f"dow_{d}"] = (X["dow"] == d).astype(float)
    doy = df["date"].dt.dayofyear
    X["season_sin"] = np.sin(2 * np.pi * doy / 365)
    X["season_cos"] = np.cos(2 * np.pi * doy / 365)
    X["trend"] = np.arange(len(df)) / 365
    X["flu_lag1"] = df["flu_index"].shift(1)
    X["flu_lag7"] = df["flu_index"].shift(7)
    X["demand_lag1"] = df[med].shift(1)
    X["demand_lag7"] = df[med].shift(7)
    X["demand_mean7"] = df[med].shift(1).rolling(7).mean()
    X["demand_mean28"] = df[med].shift(1).rolling(28).mean()
    return X
