"""
The buy-only gold EMA rule against holding gold, both with real spreads and overnight fees.

Bars are 1-hour bars built from the Pepperstone MT5 gold export (bid prices and the spread of
every minute, in broker server time; the broker's day starts at 00:00 there). Buying and
selling each pay half the spread of that bar's first minute. Every night a position is held
past 00:00 it pays the long overnight fee read from the same terminal on 2026-10-21 (79.38
points per lot, 0.0193% of the gold price), three times on Wednesday nights; that one rate is
used for every year because older rates are not in the export. Holding pays the same costs.

Two checks: the setting picked in scripts/trend_frontier.py (EMA 14/25/125) on its held-back
years, and a yearly re-pick: before each year, every setting of the scripts/traditional.py grid
is scored on all earlier years (yearly return per unit of worst drop) and the best one trades
that year. Settings are scored on 80% of the CPU cores.

Example:
    python scripts/gold_real_costs.py --output /tmp/gold-real-costs
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import traditional as t
import vectorbt as vbt

EXPORT = Path(
    "/mnt/c/LeonAI-DO/dev/TRADING/mt5-adaptive-support-resistance-research/runs/"
    "gold-data-pepperstone-20261021/xauusd_m1.parquet"
)
START = "2017-01-01"
END = "2026-08-18"
POINT = 0.01
LONG_FEE = 79.38 * POINT / 4118.65  # fraction of price per night, read at bid 4118.65
WEDNESDAY = 2
PICK = (14, 25, 125)
HELD_BACK_FROM = "2023-08-29"
FIRST_TRADED_YEAR = 2019
MIN_CHANGES = 20
DATA: dict = {}


def load():
    m1 = pd.read_parquet(EXPORT).set_index("time")
    m1 = m1.loc[START:END]  # before 2017 the export holds one bar per day
    groups = m1.resample("1h")
    bars = pd.DataFrame(
        {
            "open": groups["open"].first(),
            "close": groups["close"].last(),
            "half_spread": groups["spread"].first() * POINT / 2,
        }
    ).dropna()
    day = bars.index.normalize()
    rollover = np.zeros(len(bars))
    changed = np.asarray(day[1:] != day[:-1])
    rollover[1:][changed] = np.where(day[:-1][changed].dayofweek == WEDNESDAY, 3, 1)
    bars["rollover"] = rollover
    return bars


def growth(bars, position):
    """Per-bar growth of the account for a 0/1 position decided at each bar's open."""
    o, c = bars["open"].to_numpy(), bars["close"].to_numpy()
    cost = 1 - bars["half_spread"].to_numpy() / o
    prev = np.concatenate(([0.0], position[:-1]))
    prev_close = np.concatenate(([o[0]], c[:-1]))
    g = np.ones(len(o))
    stay = (prev == 1) & (position == 1)
    enter = (prev == 0) & (position == 1)
    leave = (prev == 1) & (position == 0)
    g[stay] = c[stay] / prev_close[stay]
    g[enter] = c[enter] / o[enter] * cost[enter]
    g[leave] = o[leave] / prev_close[leave] * cost[leave]
    fee = (prev == 1) & (bars["rollover"].to_numpy() > 0)
    g[fee] *= 1 - LONG_FEE * bars["rollover"].to_numpy()[fee]
    return g


def positions(bars, settings):
    close = bars["close"].to_numpy()
    windows = sorted({w for st in settings for w in st})
    ma = vbt.MA.run(bars["close"], window=windows, ewm=True).ma.to_numpy()
    emas = {w: ma[:, i] for i, w in enumerate(windows)}
    for st in settings:
        target = t.targets(close, emas, st)
        target[target < 0] = 0.0
        yield st, pd.Series(target).ffill().fillna(0.0).to_numpy()


def stats(g, index):
    equity = np.cumprod(g)
    drawdown = float((1 - equity / np.maximum.accumulate(equity)).max())
    total = float(equity[-1] - 1)
    days = (index[-1] - index[0]).total_seconds() / 86_400
    yearly = (1 + total) ** (365.25 / days) - 1
    return {
        "return_pct": round(total * 100, 2),
        "max_drawdown_pct": round(drawdown * 100, 2),
        "yearly_return_pct": round(yearly * 100, 2),
        "return_per_drawdown": round(yearly / drawdown, 3) if drawdown else None,
    }


def _score(settings):
    """For each setting: training score before every traded year, and that year's growth."""
    bars = DATA["bars"]
    years = np.asarray(bars.index.year)
    out = []
    for st, pos in positions(bars, settings):
        g = growth(bars, pos)
        changes = np.abs(np.diff(pos)).cumsum()
        row = {"setting": st, "train": {}, "year": {}}
        for year in range(FIRST_TRADED_YEAR, years.max() + 1):
            before = years < year
            score = stats(g[before], bars.index[before])
            if changes[before[1:]].max() >= MIN_CHANGES:
                row["train"][year] = score["return_per_drawdown"] or -1e9
            row["year"][year] = g[years == year]
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    t0 = time.time()
    bars = DATA["bars"] = load()
    years = np.asarray(bars.index.year)
    settings = [
        st for st in itertools.product(*t.GRID.values()) if st[0] < st[1] < st[2]
    ]
    workers = max(1, int(os.cpu_count() * 0.8))
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [
            r
            for chunk in pool.map(
                _score, [settings[i::workers] for i in range(workers)]
            )
            for r in chunk
        ]

    # Check 1: the frontier pick on its held-back years, against holding.
    held = bars.index >= pd.Timestamp(HELD_BACK_FROM, tz="UTC")
    pick_pos = dict(positions(bars, [PICK]))[PICK]
    pick_pos = np.where(held, pick_pos, 0.0)
    hold_pos = held.astype(float)
    for pos in (pick_pos, hold_pos):
        pos[-1] = 0.0
    check1 = {
        "pick": stats(growth(bars, pick_pos)[held], bars.index[held]),
        "hold": stats(growth(bars, hold_pos)[held], bars.index[held]),
        "pick_in_market_pct": round(float(pick_pos[held].mean()) * 100, 1),
    }

    # Check 2: re-pick each year on all earlier years, trade the pick that year.
    walk, by_year = [], {}
    for year in range(FIRST_TRADED_YEAR, years.max() + 1):
        best = max(
            (r for r in rows if year in r["train"]), key=lambda r: r["train"][year]
        )
        g = best["year"][year]
        walk.append(g)
        inside = years == year
        hold_g = growth(bars[inside], np.append(np.ones(inside.sum() - 1), 0.0))
        by_year[year] = {
            "setting": list(best["setting"]),
            "repick": stats(g, bars.index[inside]),
            "hold": stats(hold_g, bars.index[inside]),
        }
    traded = years >= FIRST_TRADED_YEAR
    hold_all = np.append(np.ones(traded.sum() - 1), 0.0)
    check2 = {
        "repick": stats(np.concatenate(walk), bars.index[traded]),
        "hold": stats(growth(bars[traded], hold_all), bars.index[traded]),
        "by_year": by_year,
    }
    summary = {
        "data": str(EXPORT),
        "bars": len(bars),
        "first_bar": str(bars.index[0]),
        "last_bar": str(bars.index[-1]),
        "median_spread_usd": round(float(bars["half_spread"].median() * 2), 3),
        "long_fee_per_night_pct": round(LONG_FEE * 100, 4),
        "settings": len(settings),
        "workers": workers,
        "seconds": round(time.time() - t0, 1),
        "check1_pick_held_back": {"from": HELD_BACK_FROM, **check1},
        "check2_yearly_repick": check2,
    }
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gold_real_costs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
