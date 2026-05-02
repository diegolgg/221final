from __future__ import annotations

import json
from pathlib import Path


OUT = Path("actual_crypto_online_risk_task.ipynb")
_CELL_COUNTER = 0


def cell_id(prefix: str) -> str:
    global _CELL_COUNTER
    _CELL_COUNTER += 1
    return f"{prefix}-{_CELL_COUNTER:03d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id("md"), "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    md(
        r"""
# Actual crypto task: online weighting for next-block stress prediction

This notebook is an actual crypto forecasting task, not a gradient-estimation proxy.

At the end of each 10-day block, we use recent crypto market features to predict whether the **next** 10-day equal-weight crypto basket return will be a stress/downside event.

The label is:

\[
y_b = \mathbf 1\{R_{b+1}^{\mathrm{mkt}} \le -7\%\}.
\]

That is a real forward-looking crypto question:

> Should we be worried about the next two weeks?

We compare three training-weighting rules under the same rolling information set:

- `uniform`: train the risk model with equal historical weights;
- `static_tass`: use an early, frozen feature model to overweight examples that look stress-informative;
- `online`: update the feature-to-stress weighting model over time, then overweight examples that currently look stress-informative.

The economic diagnostic is a simple risk-off rule: hold the equal-weight crypto basket unless predicted stress is in the model's top-risk quartile, in which case hold cash for the next block.
"""
    ),
    code(
        r"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "crypto_stage1_processed"
OUT_DIR = DATA_DIR / "actual_crypto_online_risk_task"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 221

print("project directory:", PROJECT_DIR)
print("output directory:", OUT_DIR)
"""
    ),
    md(
        r"""
## 1. Build the real crypto forecasting panel

We use the real daily return panel from the project workspace. Each non-overlapping 10-day block gets:

- current-block features, such as realized return, volatility, downside pressure, tail count, and cross-asset dispersion;
- a forward label: whether the next block loses at least 7%;
- a forward return: the next block's equal-weight crypto basket return.

The model only uses current and past block information when forecasting.
"""
    ),
    code(
        r"""
returns_daily = pd.read_pickle(DATA_DIR / "returns_daily.pkl").copy()
returns_daily["date"] = pd.to_datetime(returns_daily["date"], utc=True)

asset_cols = [c for c in returns_daily.columns if c.endswith("_ret")]
asset_names = [c.replace("_ret", "") for c in asset_cols]

L_BLOCK = 10
STRESS_RETURN_THRESHOLD = -0.07


def compound_return(r: np.ndarray) -> float:
    return float(np.prod(1.0 + r) - 1.0)


def make_crypto_blocks(df: pd.DataFrame, asset_cols: list[str], L: int = 10) -> pd.DataFrame:
    n_blocks = len(df) // L
    rows = []

    for b in range(n_blocks):
        lo, hi = b * L, (b + 1) * L
        R = df.loc[lo : hi - 1, asset_cols].to_numpy(dtype=float)
        market = R.mean(axis=1)
        cum_market = np.cumprod(1.0 + market) - 1.0

        row = {
            "block": b,
            "start_date": df.loc[lo, "date"],
            "end_date": df.loc[hi - 1, "date"],
            "market_ret": compound_return(market),
            "market_vol": float(np.std(market) * np.sqrt(L)),
            "market_drawdown": float(cum_market.min()),
            "market_abs": float(np.abs(R).mean()),
            "market_sq": float((R**2).mean()),
            "dispersion": float(np.std(R, axis=1).mean()),
            "tail_count_8pct": float((np.abs(R) > 0.08).sum()),
            "downside_market": float(np.maximum(-R, 0.0).mean()),
        }

        for name, j in zip(asset_names, range(len(asset_cols))):
            rj = R[:, j]
            row[f"{name}_ret"] = compound_return(rj)
            row[f"{name}_vol"] = float(np.std(rj) * np.sqrt(L))
            row[f"{name}_downside"] = float(np.maximum(-rj, 0.0).mean())

        rows.append(row)

    blocks = pd.DataFrame(rows)
    blocks["next_market_ret"] = blocks["market_ret"].shift(-1)
    blocks["next_block_end_date"] = blocks["end_date"].shift(-1)
    blocks["next_stress"] = (blocks["next_market_ret"] <= STRESS_RETURN_THRESHOLD).astype(float)
    blocks = blocks.iloc[:-1].reset_index(drop=True)
    return blocks


blocks = make_crypto_blocks(returns_daily, asset_cols, L=L_BLOCK)

feature_cols = [
    c
    for c in blocks.columns
    if c not in {"block", "start_date", "end_date", "next_block_end_date", "next_market_ret", "next_stress"}
]

print("assets:", asset_names)
print("date range:", blocks["start_date"].min(), "to", blocks["next_block_end_date"].max())
print("n forecast examples:", len(blocks))
print("stress threshold:", f"{STRESS_RETURN_THRESHOLD:.1%}")
print("stress rate:", f"{blocks['next_stress'].mean():.1%}")
display(blocks.head())
"""
    ),
    md(
        r"""
## 2. Lightweight logistic model utilities

We avoid external dependencies and fit ridge-regularized logistic regression directly with Newton updates.

The weighting methods differ only in the sample weights used to train the same predictive model on the same rolling history.
"""
    ),
    code(
        r"""
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic_ridge(X, y, sample_weight=None, ridge=3.0, n_iter=30):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X1 = np.column_stack([np.ones(len(X)), X])

    if sample_weight is None:
        w = np.ones(len(X))
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = w / max(float(w.mean()), 1e-8)

    beta = np.zeros(X1.shape[1])
    for _ in range(n_iter):
        p = sigmoid(X1 @ beta)
        R = w * p * (1.0 - p) + 1e-6
        grad = X1.T @ (w * (p - y))
        grad[1:] += ridge * beta[1:]

        H = X1.T @ (X1 * R[:, None])
        H[1:, 1:] += ridge * np.eye(X1.shape[1] - 1)

        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad

        beta -= step
        if np.linalg.norm(step) < 1e-6:
            break

    return beta


def predict_proba(beta, X):
    X = np.asarray(X, dtype=float)
    return sigmoid(np.column_stack([np.ones(len(X)), X]) @ beta)


def auc_score(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def max_drawdown(block_returns):
    curve = np.cumprod(1.0 + np.asarray(block_returns, dtype=float))
    peak = np.maximum.accumulate(curve)
    return float(((curve - peak) / peak).min())
"""
    ),
    md(
        r"""
## 3. Rolling forecast and risk-off backtest

At forecast time \(t\), each method sees only examples \(<t\).

The online method does not peek at the future. It updates its stress-weighting model using labels that have already been realized.

The `static_tass` method is intentionally frozen after the warmup period. This mimics a fixed precomputed targeting rule.
"""
    ),
    code(
        r"""
WARMUP = 50
ROLLING_WINDOW = 90
ONLINE_DECAY = 0.96

TARGET_RIDGE = 3.0
UTILITY_RIDGE = 5.0
WEIGHT_FLOOR = 0.25
RISK_OFF_QUANTILE = 0.75

X_raw = blocks[feature_cols].to_numpy(dtype=float)
y = blocks["next_stress"].to_numpy(dtype=float)
next_returns = blocks["next_market_ret"].to_numpy(dtype=float)

# Initial-period standardization avoids using future distribution information.
x_mean = X_raw[:WARMUP].mean(axis=0)
x_std = X_raw[:WARMUP].std(axis=0)
x_std = np.where(x_std < 1e-8, 1.0, x_std)
X = (X_raw - x_mean) / x_std

beta_static_utility = fit_logistic_ridge(X[:WARMUP], y[:WARMUP], ridge=UTILITY_RIDGE)

records = []
start_eval = WARMUP + ROLLING_WINDOW

for t in range(start_eval, len(blocks)):
    hist = np.arange(max(0, t - ROLLING_WINDOW), t)
    X_hist = X[hist]
    y_hist = y[hist]

    # Online utility model is fit on all labels available before time t with exponential decay.
    all_past = np.arange(0, t)
    decay_weights = ONLINE_DECAY ** (t - all_past)
    beta_online_utility = fit_logistic_ridge(
        X[all_past],
        y[all_past],
        sample_weight=decay_weights,
        ridge=UTILITY_RIDGE,
    )

    static_scores = predict_proba(beta_static_utility, X_hist)
    online_scores = predict_proba(beta_online_utility, X_hist)

    weights_by_method = {
        "uniform": np.ones(len(hist)),
        "static_tass": WEIGHT_FLOOR + static_scores / max(static_scores.mean(), 1e-8),
        "online": WEIGHT_FLOOR + online_scores / max(online_scores.mean(), 1e-8),
    }

    for method, weights in weights_by_method.items():
        beta = fit_logistic_ridge(X_hist, y_hist, sample_weight=weights, ridge=TARGET_RIDGE)
        p_now = float(predict_proba(beta, X[t : t + 1])[0])
        p_hist = predict_proba(beta, X_hist)
        cutoff = float(np.quantile(p_hist, RISK_OFF_QUANTILE))
        risk_off = bool(p_now >= cutoff)

        strategy_ret = 0.0 if risk_off else float(next_returns[t])

        records.append({
            "forecast_block": int(t),
            "current_block_end_date": blocks.loc[t, "end_date"],
            "forecasted_block_end_date": blocks.loc[t, "next_block_end_date"],
            "method": method,
            "predicted_stress_prob": p_now,
            "risk_cutoff": cutoff,
            "risk_off": risk_off,
            "actual_stress": float(y[t]),
            "next_market_ret": float(next_returns[t]),
            "strategy_ret": strategy_ret,
        })

results = pd.DataFrame(records)
results.to_csv(OUT_DIR / "actual_crypto_risk_predictions.csv", index=False)
blocks.to_csv(OUT_DIR / "actual_crypto_forecast_blocks.csv", index=False)

print("evaluation examples:", results["forecast_block"].nunique())
print("evaluation stress rate:", f"{results.drop_duplicates('forecast_block')['actual_stress'].mean():.1%}")
display(results.head())
"""
    ),
    md(
        r"""
## 4. Forecasting and trading diagnostics

Primary forecasting metrics:

- lower log loss is better;
- lower Brier score is better;
- higher AUC is better;
- higher stress rate in the top-risk quartile means the model concentrates alarms on genuinely bad future blocks.

Trading diagnostic:

- hold the equal-weight crypto basket unless the method flags risk-off;
- risk-off earns zero for that block;
- compare total return and max drawdown to buy-and-hold over the same evaluation period.
"""
    ),
    code(
        r"""
summary_rows = []

buy_hold_returns = results.drop_duplicates("forecast_block")["next_market_ret"].to_numpy()
buy_hold_total_return = float(np.prod(1.0 + buy_hold_returns) - 1.0)
buy_hold_max_dd = max_drawdown(buy_hold_returns)

for method, g in results.groupby("method"):
    yy = g["actual_stress"].to_numpy(dtype=float)
    pp = np.clip(g["predicted_stress_prob"].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    strat = g["strategy_ret"].to_numpy(dtype=float)

    top_mask = pp >= np.quantile(pp, 0.75)
    curve = np.cumprod(1.0 + strat)

    summary_rows.append({
        "method": method,
        "log_loss": float(-(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)).mean()),
        "brier": float(((pp - yy) ** 2).mean()),
        "auc": auc_score(yy, pp),
        "stress_rate_top_risk_quartile": float(yy[top_mask].mean()),
        "risk_off_rate": float(g["risk_off"].mean()),
        "strategy_total_return": float(curve[-1] - 1.0),
        "strategy_max_drawdown": max_drawdown(strat),
        "buy_hold_total_return": buy_hold_total_return,
        "buy_hold_max_drawdown": buy_hold_max_dd,
    })

summary = pd.DataFrame(summary_rows).sort_values("log_loss").reset_index(drop=True)
summary.to_csv(OUT_DIR / "actual_crypto_risk_summary.csv", index=False)
display(summary)
"""
    ),
    md(
        r"""
## 5. Visualize predictions and realized stress

The vertical markers indicate realized future stress events. The online curve should be read as a real-time risk score, not a price forecast.
"""
    ),
    code(
        r"""
plot_df = results.copy()
plot_df["forecasted_block_end_date"] = pd.to_datetime(plot_df["forecasted_block_end_date"], utc=True)

fig, ax = plt.subplots(figsize=(13, 5))
for method, g in plot_df.groupby("method"):
    s = g.set_index("forecasted_block_end_date")["predicted_stress_prob"].rolling(4, min_periods=1).mean()
    lw = 2.4 if method == "online" else 1.7
    ax.plot(s.index, s.values, label=method, lw=lw)

stress_dates = plot_df.drop_duplicates("forecast_block")
stress_dates = stress_dates.loc[stress_dates["actual_stress"] == 1, "forecasted_block_end_date"]
for d in stress_dates:
    ax.axvline(d, color="tab:red", alpha=0.12, lw=1)

ax.set_title("Predicted next-block crypto stress probability")
ax.set_ylabel("predicted probability")
ax.set_xlabel("forecasted block end date")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "actual_crypto_predicted_stress_probability.png", dpi=180)
plt.show()
"""
    ),
    md(
        r"""
## 6. Risk-off equity curves

This is the money-flavored diagnostic. It is deliberately simple: when the model says the next block is in its top-risk quartile, hold cash; otherwise hold the equal-weight crypto basket.

Do not overinterpret this as a production trading system. The real point is whether the risk model identifies bad future blocks better than uniform or frozen-static weighting.
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(13, 5))

date_index = (
    plot_df.drop_duplicates("forecast_block")
    .sort_values("forecast_block")["forecasted_block_end_date"]
    .to_numpy()
)
buy_hold_curve = np.cumprod(1.0 + buy_hold_returns)
ax.plot(date_index, buy_hold_curve, label="buy_hold", color="black", lw=2.0, alpha=0.75)

for method, g in plot_df.sort_values("forecast_block").groupby("method"):
    curve = np.cumprod(1.0 + g["strategy_ret"].to_numpy(dtype=float))
    lw = 2.5 if method == "online" else 1.8
    ax.plot(g["forecasted_block_end_date"], curve, label=method, lw=lw)

ax.set_title("Risk-off strategy equity curves")
ax.set_ylabel("growth of $1")
ax.set_xlabel("forecasted block end date")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_DIR / "actual_crypto_risk_off_equity_curves.png", dpi=180)
plt.show()
"""
    ),
    md(
        r"""
## 7. Interpretation

This is now an actual crypto task:

- input: current and historical crypto returns;
- target: whether the next 10-day equal-weight crypto basket return is worse than \(-7\%\);
- decision: risk-on versus risk-off;
- comparison: equal weighting, frozen static targeted weighting, and online adaptive weighting.

The online method is useful here when the feature pattern associated with future downside changes over time. That is a plausible crypto-market phenomenon: 2021-style broad volatility, 2022-style deleveraging, and later cross-asset dispersion are not the same regime.
"""
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "stat221-ps5",
            "language": "python",
            "name": "stat221-ps5",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT.resolve()}")
