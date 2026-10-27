"""
The 3-EMA crossover as 5-minute Rise/Fall binary options on gold (XAUUSD), 80% payout.

A 1-minute close crossing above all three EMAs buys Rise (CALL), crossing below all three buys
Fall (PUT), as in scripts/traditional.py. The contract starts at the next minute's open and
settles at the close of the fifth minute; a tie loses. Contracts whose five minutes are not all
in the data (weekends, gaps) are skipped. A win pays 80% of the stake and a loss costs the
stake, so the break-even win rate is 1 / 1.8 = 55.56%. Both directions and Rise only are tried
for the EMA grid of scripts/traditional.py on the first 70% of Aug 2016 to Aug 2026 (bid prices
from scripts/histdata_m1.py); the 10 settings with the highest trust score are then tested once
on the last 30%, as scripts/ema_frontier.py did for Volatility 75. Money: 10,000 USD start and
each stake 0.05% of the balance, without the 0.35 USD minimum or the 5-open limit.

Example:
    python scripts/gold_binary.py --data ~/data/histdata --output /tmp/gold-binary
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import traditional as t
import traditional_markets as m
import vectorbt as vbt

PAYOUT = 0.80
BREAK_EVEN = 1 / (1 + PAYOUT)
EXPIRY_MIN = 5
STAKE_FRACTION = 0.0005
DATA: dict = {}


def outcomes(bars):
    """For a contract starting at bar j: valid, rose, fell (close of bar j+4 against open j)."""
    minutes = np.asarray((bars.index - bars.index[0]) // pd.Timedelta(minutes=1))
    n = len(bars) - (EXPIRY_MIN - 1)
    end = np.arange(n) + EXPIRY_MIN - 1
    valid = np.zeros(len(bars), dtype=bool)
    rose = np.zeros(len(bars), dtype=bool)
    fell = np.zeros(len(bars), dtype=bool)
    opens, closes = bars["open"].to_numpy(), bars["close"].to_numpy()
    valid[:n] = minutes[end] - minutes[:n] == EXPIRY_MIN - 1
    rose[:n] = closes[end] > opens[:n]
    fell[:n] = closes[end] < opens[:n]
    return valid, rose, fell


def score(won, days):
    n = int(won.size)
    wins = int(won.sum())
    rate = wins / n if n else 0.0
    equity = 10_000.0 * float(
        np.prod(np.where(won, 1 + STAKE_FRACTION * PAYOUT, 1 - STAKE_FRACTION))
    )
    return {
        "contracts": n,
        "per_day": round(n / days, 2),
        "win_rate": round(rate, 5),
        "trust": round(
            (rate - BREAK_EVEN) / math.sqrt(BREAK_EVEN * (1 - BREAK_EVEN) / n), 2
        )
        if n
        else 0.0,
        "return_per_stake_pct": round((rate * (1 + PAYOUT) - 1) * 100, 2),
        "net_usd": round(equity - 10_000.0, 2),
    }


def evaluate(settings, part, rise_only):
    bars = DATA["bars"]
    valid, rose, fell = DATA["outcomes"]
    close = bars["close"].to_numpy()
    windows = sorted({w for st in settings for w in st})
    ma = vbt.MA.run(bars["close"], window=windows, ewm=True).ma.to_numpy()
    emas = {w: ma[:, i] for i, w in enumerate(windows)}
    inside = np.zeros(len(bars), dtype=bool)
    inside[part[0] : part[1]] = True
    days = (bars.index[part[1] - 1] - bars.index[part[0]]).total_seconds() / 86_400
    out = []
    for st in settings:
        target = t.targets(close, emas, st)
        call = (target == 1) & valid & inside
        put = (target == -1) & valid & inside & (not rise_only)
        won = np.concatenate([rose[call], fell[put]])
        out.append({"setting": list(st), "rise_only": rise_only, **score(won, days)})
    return out


def _search(args):
    return evaluate(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    t0 = time.time()
    DATA["bars"] = m.load(args.data, "XAUUSD")
    DATA["outcomes"] = outcomes(DATA["bars"])
    split = int(len(DATA["bars"]) * 0.7)
    settings = [
        st for st in itertools.product(*t.GRID.values()) if st[0] < st[1] < st[2]
    ]
    workers = max(1, int(os.cpu_count() * 0.8))
    per = max(1, workers // 2)
    tasks = [
        (settings[i::per], (0, split), rise_only)
        for rise_only in (False, True)
        for i in range(per)
    ]
    with mp.get_context("fork").Pool(workers) as pool:
        rows = [r for chunk in pool.map(_search, tasks) for r in chunk]
    top = sorted(rows, key=lambda r: r["trust"], reverse=True)[:10]
    held = []
    for r in top:
        held += evaluate(
            [tuple(r["setting"])], (split, len(DATA["bars"])), r["rise_only"]
        )
    default = [
        evaluate([t.DEFAULT], part, False)[0]
        for part in ((0, split), (split, len(DATA["bars"])))
    ]
    summary = {
        "payout": PAYOUT,
        "break_even_win_rate": round(BREAK_EVEN, 5),
        "expiry_minutes": EXPIRY_MIN,
        "settings_tested": len(rows),
        "luck_trust_level": round(math.sqrt(2 * math.log(len(rows))), 2),
        "held_back_from": str(DATA["bars"].index[split]),
        "seconds": round(time.time() - t0, 1),
        "search_above_break_even": sum(r["win_rate"] > BREAK_EVEN for r in rows),
        "search_best_win_rate": max(r["win_rate"] for r in rows),
        "search_median_win_rate": float(np.median([r["win_rate"] for r in rows])),
        "top_by_trust": top,
        "top_held_back": held,
        "default_9_21_200": {"search": default[0], "held_back": default[1]},
    }
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "gold_binary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (folder / "points.json").write_text(json.dumps(rows), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if "top" not in k}, indent=2))


if __name__ == "__main__":
    main()
