"""Predict-then-optimize: does the most accurate forecast give the best ordering decisions?

Timeline:  2023 = training,  2024 = rolling validation (learn forecast errors),  2025 = test year.
Usage:  python src/run.py     (from the project root). Outputs go to results/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CABINET_CAPACITY, MEDICINES, make_data
from forecast import fit_predict
from optimize import realized_cost, saa_order, scale_to_capacity

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
N_SCENARIOS = 200
METHODS = ["Naive (last week)", "Linear regression", "Gradient boosting", "Quantile gradient boosting",
           "Stochastic LP (SAA)", "Perfect foresight"]


def main():
    df = make_data()
    names = [m.name for m in MEDICINES]
    h = np.array([m.holding for m in MEDICINES])
    p = np.array([m.shortage for m in MEDICINES])
    vol = np.array([m.volume for m in MEDICINES])
    crit = p / (p + h)
    year = df["date"].dt.year

    # 1) Validation: train on 2023, forecast 2024 -> learn how wrong the forecast tends to be
    print("Rolling validation forecasts for 2024 (refit every quarter) ...")
    # Standardized errors z = (actual - forecast) / sqrt(forecast): count-data errors grow with the square root
    # of the level, so z can be reused at any demand level. One row = one day's joint errors (keeps correlation).
    # Rolling-origin validation: refit at the start of each quarter of 2024 on all data before it, predict that quarter.
    quarter_starts = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01"])

    def rolling_errors(med):
        z = []
        for a, b in zip(quarter_starts[:-1], quarter_starts[1:]):
            v = fit_predict(df, med, df["date"] < a, (df["date"] >= a) & (df["date"] < b))
            z.append(((v["actual"] - v["gbm"]) / np.sqrt(v["gbm"].clip(lower=1.0))).to_numpy())
        return np.concatenate(z)

    errors = np.column_stack([rolling_errors(m) for m in names])

    # 2) Test: train on 2023-2024, forecast 2025
    print("Test forecasts (train 2023-2024, predict 2025) ...")
    preds = {m: fit_predict(df, m, year <= 2024, year == 2025, quantile=float(c)) for m, c in zip(names, crit)}
    actual = np.column_stack([preds[m]["actual"] for m in names])
    fc = {k: np.column_stack([preds[m][k] for m in names]) for k in ["naive", "linear", "gbm", "gbm_quantile"]}

    # 3) Decisions for every day of 2025
    print("Optimizing daily orders for 2025 ...")
    rng = np.random.default_rng(0)
    days = len(actual)
    orders = {k: np.zeros_like(actual, dtype=float) for k in METHODS}
    for t in range(days):
        orders["Naive (last week)"][t] = scale_to_capacity(fc["naive"][t], vol, CABINET_CAPACITY)
        orders["Linear regression"][t] = scale_to_capacity(fc["linear"][t], vol, CABINET_CAPACITY)
        orders["Gradient boosting"][t] = scale_to_capacity(fc["gbm"][t], vol, CABINET_CAPACITY)
        orders["Quantile gradient boosting"][t] = scale_to_capacity(fc["gbm_quantile"][t], vol, CABINET_CAPACITY)
        base = fc["gbm"][t]
        scen = np.clip(base + errors[rng.integers(0, len(errors), N_SCENARIOS)] * np.sqrt(np.maximum(base, 1.0)), 0, None)
        orders["Stochastic LP (SAA)"][t] = saa_order(scen, h, p, vol, CABINET_CAPACITY)
        orders["Perfect foresight"][t] = saa_order(actual[t][None, :], h, p, vol, CABINET_CAPACITY)

    # 4) Evaluate forecast accuracy and decision quality
    test_dates = df.loc[year == 2025, "date"].reset_index(drop=True)
    flu_season = test_dates.dt.month.isin([1, 2, 3]).to_numpy()
    mae = {"Naive (last week)": "naive", "Linear regression": "linear", "Gradient boosting": "gbm",
           "Quantile gradient boosting": "gbm_quantile"}
    rows, daily = [], {}
    for k in METHODS:
        res = [realized_cost(orders[k][t], actual[t], h, p) for t in range(days)]
        cost = np.array([r[0] for r in res])
        short = np.array([r[1] for r in res])
        daily[k] = cost
        rows.append({
            "method": k,
            "forecast_MAE": np.abs(fc[mae[k]] - actual).mean() if k in mae else np.nan,
            "annual_cost_eur": cost.sum(),
            "cost_flu_season_jan_mar": cost[flu_season].sum(),
            "cost_rest_of_year": cost[~flu_season].sum(),
            "fill_rate": 1 - short.sum() / actual.sum(),
            "days_cabinet_full": int((orders[k] @ vol > CABINET_CAPACITY - 1e-6).sum()),
            "share_orders_covering_demand": float((orders[k] >= actual - 1e-9).mean()),
        })
    summary = pd.DataFrame(rows).set_index("method")
    best_plugin = summary.loc["Gradient boosting", "annual_cost_eur"]
    summary["saving_vs_gbm_point"] = 1 - summary["annual_cost_eur"] / best_plugin
    summary.to_csv(OUT / "summary_2025.csv")

    per_med = pd.DataFrame({m: {"MAE_gbm": np.abs(fc["gbm"][:, i] - actual[:, i]).mean(),
                                "critical_ratio": crit[i],
                                "mean_demand": actual[:, i].mean()} for i, m in enumerate(names)}).T
    per_med.to_csv(OUT / "per_medicine.csv")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n=== 2025 test year: forecast accuracy vs. decision cost ===")
    print(summary.to_string(float_format=lambda v: f"{v:,.3f}"))
    print("\n=== Per medicine ===")
    print(per_med.to_string(float_format=lambda v: f"{v:,.2f}"))

    plot_accuracy_vs_cost(summary)
    plot_cumulative(test_dates, daily, df.loc[year == 2025, "flu_index"].to_numpy())
    plot_forecast_example(test_dates, preds["Antiviral oral"], orders["Stochastic LP (SAA)"][:, 1])
    print(f"\nResults written to {OUT}")


def plot_accuracy_vs_cost(summary):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pts = summary.dropna(subset=["forecast_MAE"])
    for name, r in pts.iterrows():
        ax.scatter(r["forecast_MAE"], r["annual_cost_eur"], s=70)
        ax.annotate(name, (r["forecast_MAE"], r["annual_cost_eur"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    saa = summary.loc["Stochastic LP (SAA)", "annual_cost_eur"]
    pf = summary.loc["Perfect foresight", "annual_cost_eur"]
    ax.axhline(saa, color="#2e7d32", ls="--", lw=1)
    ax.text(ax.get_xlim()[0], saa, "  Stochastic LP (SAA)", va="bottom", fontsize=8, color="#2e7d32")
    ax.axhline(pf, color="#777777", ls=":", lw=1)
    ax.text(ax.get_xlim()[0], pf, "  Perfect foresight", va="bottom", fontsize=8, color="#777777")
    ax.set_xlabel("Forecast error (MAE, packs per medicine per day)")
    ax.set_ylabel("Annual cost of ordering decisions [EUR]")
    ax.set_title("A better forecast is not the same as a better decision")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


def plot_cumulative(dates, daily, flu):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k in METHODS:
        ax.plot(dates, np.cumsum(daily[k]), label=k, lw=1.6)
    b = ax.twinx()
    b.fill_between(dates, flu, color="#999999", alpha=0.2)
    b.set_ylabel("Flu index (shaded)")
    ax.set_ylabel("Cumulative cost in 2025 [EUR]")
    ax.set_title("Cumulative ordering cost over the 2025 test year")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cumulative_cost.png", dpi=150)
    plt.close(fig)


def plot_forecast_example(dates, pred, saa_q):
    sel = slice(0, 120)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(dates[sel], pred["actual"].to_numpy()[sel], color="black", lw=1, label="Actual demand")
    ax.plot(dates[sel], pred["gbm"].to_numpy()[sel], color="#2e86ab", lw=1.4, label="Gradient boosting forecast")
    ax.plot(dates[sel], pred["gbm_quantile"].to_numpy()[sel], color="#f18f01", lw=1.2, ls="--",
            label="Quantile forecast (critical ratio 0.90)")
    ax.plot(dates[sel], saa_q[sel], color="#2e7d32", lw=1.2, label="Stochastic LP order")
    ax.set_ylabel("Packs per day")
    ax.set_title("Antiviral oral, January–April 2025 (flu season)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "forecast_example_antiviral.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
