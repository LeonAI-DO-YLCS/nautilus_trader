"""
Independent re-test of the frozen Step Index reversal set from the owner's binary options lab.

Rules, as in the lab's spec and MT5 robot (StepReversalSet.mq5), each trading 0.1 lot (1 point
of price = 1 USD), on completed bars with orders at the next bar's open:
  1. 1-hour RSI(7), Wilder smoothing: buy below 20, sell above 80, exit when RSI crosses 50.
  2. The same with 25 and 75.
  3. 4-hour reversal: hold against the change over the last 120 bars, always in the market.
  4. Daily Bollinger 20 / 1.5 (sample standard deviation): buy below the lower band, sell above
     the upper band, exit when the close crosses the average.
An RSI or Bollinger exit and a new entry may happen on the same bar, as in the robot.

Costs: each side pays half the measured spread of the bar's first minute (the lab used a fixed
0.2 points round trip). Every weekday night (Monday to Friday) a held position pays 1% a year
of the price, 1/360 per night, as in the lab's spec. The ledger is plain numpy because
vectorbt's portfolios have no overnight fee.

Periods match the lab: rules picked on 2019-05-28 to 2021-12-31, checked on 2022 to 2023 and
2024 to the end of the data. The script also reports every calendar year, a cost stress (both
costs doubled) and nearby settings of each rule, run on 80% of the CPU cores.

Example:
    python scripts/step_reversal.py --bars ~/data/step/step_m1.parquet --output /tmp/step
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOT_POINT_USD = 1.0  # 0.1 lot: 1 point of price = 1 USD
SWAP_YEARLY = 0.01
PERIODS = {
    "select 2019-2021": ("2019-05-28", "2022-01-01"),
    "test 2022-2023": ("2022-01-01", "2024-01-01"),
    "test 2024-2026": ("2024-01-01", "2027-01-01"),
}
FROZEN = {
    "H1 RSI 7 levels 20/80": ("rsi", "1h", (7, 20.0)),
    "H1 RSI 7 levels 25/75": ("rsi", "1h", (7, 25.0)),
    "H4 reversal 120": ("reversal", "4h", (120,)),
    "D1 Bollinger 20/1.5": ("bollinger", "1D", (20, 1.5)),
}
NEARBY = {
    "rsi": [(n, lv) for n in (5, 7, 9, 11) for lv in (15.0, 20.0, 25.0, 30.0)],
    "reversal": [(n,) for n in (60, 90, 120, 150, 180)],
    "bollinger": [(n, k) for n in (15, 20, 25) for k in (1.25, 1.5, 2.0)],
}
FRAME_OF = {"rsi": "1h", "reversal": "4h", "bollinger": "1D"}
BARS: dict = {}


def resample(m1, rule):
    g = m1.resample(rule)
    return pd.DataFrame(
        {
            "open": g["open"].first(),
            "close": g["close"].last(),
            "spread": g["spread"].first(),
        }
    ).dropna()


def rsi_positions(close, period, level):
    diff = np.diff(close, prepend=close[0])
    alpha = 1 / period
    gain = pd.Series(np.clip(diff, 0, None)).ewm(alpha=alpha, adjust=False).mean()
    loss = pd.Series(np.clip(-diff, 0, None)).ewm(alpha=alpha, adjust=False).mean()
    rsi = (100 - 100 / (1 + gain / loss)).to_numpy()
    want = np.zeros(close.size)
    p = 0
    for i in range(period, close.size):
        r = rsi[i]
        if (p == 1 and r > 50) or (p == -1 and r < 50):
            p = 0
        if p == 0:
            p = 1 if r < level else (-1 if r > 100 - level else 0)
        want[i] = p
    return want


def reversal_positions(close, bars_back):
    want = np.zeros(close.size)
    change = close[bars_back:] - close[:-bars_back]
    want[bars_back:] = -np.sign(change)
    # A zero change goes flat, as in the robot.
    return want


def bollinger_positions(close, period, k):
    s = pd.Series(close)
    z = ((s - s.rolling(period).mean()) / s.rolling(period).std(ddof=1)).to_numpy()
    want = np.zeros(close.size)
    p = 0
    for i in range(period - 1, close.size):
        if not np.isfinite(z[i]):
            want[i] = p
            continue
        if (p == 1 and z[i] > 0) or (p == -1 and z[i] < 0):
            p = 0
        if p == 0:
            p = 1 if z[i] < -k else (-1 if z[i] > k else 0)
        want[i] = p
    return want


POSITIONS = {
    "rsi": rsi_positions,
    "reversal": reversal_positions,
    "bollinger": bollinger_positions,
}


def ledger(bars, want, cost_factor=1.0):
    """Per-bar P/L in points: the position wanted at bar i's close is held from bar i+1's open."""
    o = bars["open"].to_numpy()
    held = np.concatenate(([0.0], want[:-1]))
    held[-1] = 0.0  # flat at the end of the data
    nxt = np.append(o[1:], bars["close"].to_numpy()[-1])
    move = held * (nxt - o)
    spread = (
        cost_factor * bars["spread"].to_numpy() / 2 * np.abs(np.diff(held, prepend=0.0))
    )
    day = bars.index.normalize()
    new_day = np.concatenate(([False], np.asarray(day[1:] != day[:-1])))
    weekday_night = new_day & np.concatenate(
        ([False], np.asarray(day[:-1].dayofweek < 5))
    )
    prev_held = np.concatenate(([0.0], held[:-1]))
    swap = cost_factor * weekday_night * np.abs(prev_held) * o * SWAP_YEARLY / 360
    entries = (held != 0) & (held != prev_held)
    return pd.DataFrame(
        {
            "pnl": (move - spread - swap) * LOT_POINT_USD,
            "spread": spread,
            "swap": swap,
            "entries": entries,
        },
        index=bars.index,
    )


def summary(book):
    daily = book["pnl"].resample("1D").sum()
    equity = daily.cumsum()
    months = book["pnl"].resample("MS").sum()
    return {
        "net_points": round(float(book["pnl"].sum()), 1),
        "spread_points": round(float(book["spread"].sum()), 1),
        "swap_points": round(float(book["swap"].sum()), 1),
        "trades": int(book["entries"].sum()),
        "max_drawdown_points": round(float((equity.cummax() - equity).max()), 1),
        "daily_sharpe_365": round(float(daily.mean() / daily.std() * np.sqrt(365)), 2)
        if daily.std() > 0
        else 0.0,
        "profitable_months": f"{int((months > 0).sum())} of {len(months)}",
    }


def by_period(book):
    out = {}
    for name, (a, b) in PERIODS.items():
        part = book.loc[a:b]
        part = part[part.index < pd.Timestamp(b, tz="UTC")]
        out[name] = summary(part)
    return out


def run(kind, params, cost_factor=1.0):
    bars = BARS[FRAME_OF[kind]]
    want = POSITIONS[kind](bars["close"].to_numpy(), *params)
    return ledger(bars, want, cost_factor)


def _nearby(args):
    kind, params = args
    periods = by_period(run(kind, params))
    return kind, list(params), {k: v["net_points"] for k, v in periods.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--bars", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    t0 = time.time()
    m1 = pd.read_parquet(Path(args.bars).expanduser()).set_index("time")
    for frame in ("1h", "4h", "1D"):
        BARS[frame] = resample(m1, frame)
    result = {
        "bars_file": args.bars,
        "first_minute": str(m1.index[0]),
        "last_minute": str(m1.index[-1]),
        "median_spread_points": float(m1["spread"].median()),
        "rules": {},
        "set": {},
        "by_year": {},
        "cost_stress_double": {},
    }
    books, stressed = {}, {}
    for name, (kind, frame, params) in FROZEN.items():
        books[name] = run(kind, params)
        stressed[name] = run(kind, params, cost_factor=2.0)
        result["rules"][name] = by_period(books[name])
    combined = (
        pd.concat([b["pnl"] for b in books.values()], axis=1).fillna(0).sum(axis=1)
    )
    entries = (
        pd.concat([b["entries"] for b in books.values()], axis=1)
        .fillna(False)
        .sum(axis=1)
    )
    spreads = (
        pd.concat([b["spread"] for b in books.values()], axis=1).fillna(0).sum(axis=1)
    )
    swaps = pd.concat([b["swap"] for b in books.values()], axis=1).fillna(0).sum(axis=1)
    set_book = pd.DataFrame(
        {"pnl": combined, "spread": spreads, "swap": swaps, "entries": entries}
    )
    result["set"] = by_period(set_book)
    for year, part in set_book.groupby(set_book.index.year):
        result["by_year"][int(year)] = summary(part)
    stress = pd.DataFrame(
        {
            "pnl": pd.concat([b["pnl"] for b in stressed.values()], axis=1)
            .fillna(0)
            .sum(axis=1),
            "spread": 0.0,
            "swap": 0.0,
            "entries": entries,
        }
    )
    result["cost_stress_double"] = by_period(stress)
    tasks = [(kind, p) for kind, params in NEARBY.items() for p in params]
    workers = max(1, min(len(tasks), int(os.cpu_count() * 0.8)))
    with mp.get_context("fork").Pool(workers) as pool:
        nearby = pool.map(_nearby, tasks)
    result["nearby"] = {
        kind: {
            "settings": sum(k == kind for k, _, _ in nearby),
            "positive_in_all_three": sum(
                k == kind and all(v > 0 for v in nets.values()) for k, _, nets in nearby
            ),
            "positive_in_both_tests": sum(
                k == kind
                and all(v > 0 for p, v in nets.items() if p.startswith("test"))
                for k, _, nets in nearby
            ),
            "rows": [{"params": p, **nets} for k, p, nets in nearby if k == kind],
        }
        for kind in NEARBY
    }
    result["seconds"] = round(time.time() - t0, 1)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "step_reversal.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    set_book["pnl"].resample("1D").sum().cumsum().rename("equity_points").to_csv(
        folder / "set_equity_daily.csv"
    )
    print(json.dumps({k: result[k] for k in ("rules", "set", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
