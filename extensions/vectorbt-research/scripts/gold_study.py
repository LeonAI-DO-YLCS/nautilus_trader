"""
How gold behaves, before mining rules: hours, sessions, weekdays, trend or bounce-back, costs.

Uses the Pepperstone MT5 gold export (1-minute bid bars and spread, 2017 onward). Broker
server time is New York time + 7 hours all year, so the broker day starts at 17:00 New York.
Each measure is given for the tuning years (2017-2022) and the later years (2023 onward), so
a pattern can be judged by whether it holds in both.

  - hours: mean move, its t-value, share of rising hours, median spread;
  - sessions: Asia (broker 01-09), London (09-15), New York (15-24) moves per day;
  - weekdays: mean daily move;
  - trend or bounce-back: variance ratios of hourly moves (above 1 = trend, below 1 =
    bounce-back), and next-day / next-week moves after rising or falling past N days;
  - reopen check: buying at broker 01:MM (ask = bid + that minute's spread) and selling
    at 05:00 bid, per entry minute, to see whether a gain needs the first minute after
    the daily reopen.

Example:
    python scripts/gold_study.py --output /tmp/gold-study
"""

import argparse
import json
from pathlib import Path

import gold_real_costs as grc
import numpy as np
import pandas as pd

PERIODS = {
    "2017-2022": ("2017-01-01", "2023-01-01"),
    "2023-2026": ("2023-01-01", "2027-01-01"),
}
SESSIONS = {"asia": (1, 9), "london": (9, 15), "new_york": (15, 24)}
LOOKBACKS = (1, 5, 20, 60, 120)
RATIOS = (2, 4, 8, 24, 120, 480)


def t_value(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size))) if x.size > 2 else 0.0


def variance_ratio(r, q):
    c = np.concatenate(([0.0], np.cumsum(r)))
    many = c[q::q] - c[:-q:q][: c[q::q].size]
    return float(many.var() / (q * r.var()))


def study(hourly, daily):
    out = {}
    r = np.log(hourly["close"]).diff().dropna()
    hours = []
    for h, x in r.groupby(r.index.hour):
        hours.append(
            {
                "broker_hour": int(h),
                "new_york_hour": int((h - 7) % 24),
                "mean_bp": round(float(x.mean() * 1e4), 2),
                "t": round(t_value(x), 2),
                "rising_share": round(float((x > 0).mean()), 3),
                "median_spread_points": float(
                    hourly["spread"][hourly.index.hour == h].median()
                ),
            }
        )
    out["hours"] = hours
    day = r.index.normalize()
    sessions = {}
    for name, (lo, hi) in SESSIONS.items():
        x = (
            r[(r.index.hour >= lo) & (r.index.hour < hi)]
            .groupby(day[(r.index.hour >= lo) & (r.index.hour < hi)])
            .sum()
        )
        sessions[name] = {
            "mean_bp_per_day": round(float(x.mean() * 1e4), 2),
            "t": round(t_value(x), 2),
            "rising_share": round(float((x > 0).mean()), 3),
        }
    out["sessions"] = sessions
    d = np.log(daily["close"]).diff().dropna()
    out["weekdays"] = {
        str(k): {"mean_bp": round(float(x.mean() * 1e4), 1), "t": round(t_value(x), 2)}
        for k, x in d.groupby(d.index.day_of_week)
    }
    out["variance_ratio_hourly"] = {
        str(q): round(variance_ratio(r.to_numpy(), q), 3) for q in RATIOS
    }
    follow = {}
    dc = np.log(daily["close"])
    nxt1 = dc.shift(-1) - dc
    nxt5 = dc.shift(-5) - dc
    for n in LOOKBACKS:
        past = dc - dc.shift(n)
        ok = past.notna() & nxt5.notna()
        up = ok & (past > 0)
        dn = ok & (past < 0)
        follow[str(n)] = {
            "next_day_after_rise_bp": round(float(nxt1[up].mean() * 1e4), 1),
            "next_day_after_fall_bp": round(float(nxt1[dn].mean() * 1e4), 1),
            "next_week_after_rise_bp": round(float(nxt5[up].mean() * 1e4), 1),
            "next_week_after_fall_bp": round(float(nxt5[dn].mean() * 1e4), 1),
            "t_next_day_rise_minus_fall": round(
                float(
                    (nxt1[up].mean() - nxt1[dn].mean())
                    / np.sqrt(nxt1[up].var() / up.sum() + nxt1[dn].var() / dn.sum())
                ),
                2,
            ),
        }
    out["after_past_days"] = follow
    out["daily_volatility_pct"] = round(float(d.std() * 100), 3)
    out["volatility_clustering_lag1"] = round(float(d.abs().autocorr(1)), 3)
    out["mean_daily_bp"] = round(float(d.mean() * 1e4), 2)
    out["days"] = int(d.size)
    return out


def reopen_check(m1, minutes=(0, 1, 2, 5, 10, 15, 30, 45)):
    """Mean move in bp from buying at 01:MM to selling at 05:00, per entry minute and period."""
    day = m1.index.normalize()
    exit_ = m1["open"][(m1.index.hour == 5) & (m1.index.minute == 0)]
    exit_.index = exit_.index.normalize()
    out = []
    for mi in minutes:
        pick = (m1.index.hour == 1) & (m1.index.minute == mi)
        entry = m1.loc[pick, ["open", "spread"]].set_index(day[pick])
        j = entry.join(exit_.rename("exit"), how="inner")
        r = (j["exit"] - (j["open"] + j["spread"] * 0.01)) / j["open"] * 1e4
        for name, (lo, hi) in PERIODS.items():
            x = r.loc[lo:hi]
            out.append(
                {
                    "entry_minute": mi,
                    "years": name,
                    "days": int(x.size),
                    "mean_bp": round(float(x.mean()), 2),
                    "t": round(t_value(x), 2),
                }
            )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    m1 = pd.read_parquet(grc.EXPORT).set_index("time").loc["2017-01-01":]
    hourly = m1.resample("1h").agg({"close": "last", "spread": "median"}).dropna()
    daily = m1.resample("1D").agg({"close": "last"}).dropna()
    out = {}
    for name, (lo, hi) in PERIODS.items():
        out[name] = study(hourly.loc[lo:hi], daily.loc[lo:hi])
    out["reopen_check"] = reopen_check(m1)
    out["last_bar"] = str(m1.index[-1])
    (args.output / "gold_study.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
