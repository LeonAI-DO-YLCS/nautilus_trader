"""
The 3-EMA crossover as traditional trading on EURUSD and gold, on 1-hour, 4-hour and daily bars.

Same rule, grid and costs as scripts/traditional_markets.py (stop and reverse when the close
crosses all three EMAs, fill at the next bar's open, half a fixed spread per side, swap not
included), on bars built from the scripts/histdata_m1.py files. Slow bars are few, so each
timeframe uses the whole 10-year span: settings are searched on the first 70% and the best is
tested once on the last 30%. The best and 9/21/200 are then run on every calendar year, with
EMAs warmed up on the whole span. Bars start at 22:00 UTC, the usual forex day start.

Example:
    python scripts/traditional_slow.py --data ~/data/histdata --output /tmp/slow
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
import traditional_markets as m

FRAMES = {"1h": None, "4h": "2h", "1D": "22h"}
MIN_TRADES = 20
DATA: dict = {}


def resample(bars, rule, offset):
    # pandas ignores a daily offset on a time-zone-aware index, so group on naive UTC.
    groups = bars.tz_localize(None).resample(rule, offset=offset)
    out = pd.DataFrame(
        {
            "open": groups["open"].first(),
            "close": groups["close"].last(),
            "half_spread": groups["half_spread"].first(),
        }
    ).dropna()
    return out.tz_localize("UTC")


def hold_pct(bars, start, stop):
    """Buy-and-hold return over bars[start:stop], for comparison."""
    return round((bars["close"].iloc[stop - 1] / bars["open"].iloc[start] - 1) * 100, 3)


def _search(args):
    key, settings, part, fees = args
    return key, t.backtest(DATA[key], settings, part, fees)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    t0 = time.time()
    for pair in m.SPREAD:
        minute = m.load(args.data, pair)
        for frame, offset in FRAMES.items():
            DATA[(pair, frame)] = resample(minute, frame, offset)
    settings = [s for s in itertools.product(*t.GRID.values()) if s[0] < s[1] < s[2]]
    workers = max(1, int(os.cpu_count() * 0.8))
    split = {key: int(len(bars) * 0.7) for key, bars in DATA.items()}
    chunks = [settings[i :: workers // len(DATA)] for i in range(workers // len(DATA))]
    tasks = [
        (key, c, (0, split[key]), fees)
        for fees in (True, False)
        for key in DATA
        for c in chunks
    ]
    with mp.get_context("fork").Pool(workers) as pool:
        results = pool.map(_search, tasks)
    summary = {
        "costs": "half of a fixed spread per side; swap not included",
        "spread": m.SPREAD,
        "years_span": m.SPAN,
        "settings": len(settings),
        "min_trades": MIN_TRADES,
        "workers": workers,
    }
    for key, bars in DATA.items():
        net = [
            r
            for (k, rows), task in zip(results, tasks)
            if k == key and task[3]
            for r in rows
        ]
        gross = [
            r
            for (k, rows), task in zip(results, tasks)
            if k == key and not task[3]
            for r in rows
        ]
        ranked = sorted(
            (r for r in net if r["trades"] >= MIN_TRADES),
            key=lambda r: r["sharpe"],
            reverse=True,
        )
        best = tuple(ranked[0]["setting"])
        held = t.backtest(bars, [best, t.DEFAULT], (split[key], len(bars)))
        years = np.asarray(bars.index.year)
        by_year = {}
        for year in sorted(set(years)):
            part = (
                int(np.argmax(years == year)),
                int(len(years) - np.argmax(years[::-1] == year)),
            )
            after = t.backtest(bars, [best, t.DEFAULT], part)
            before = t.backtest(bars, [best], part, fees=False)
            by_year[int(year)] = {
                "best": after[0],
                "default": after[1],
                "best_before_costs": before[0],
                "buy_and_hold_pct": hold_pct(bars, *part),
            }
        summary[f"{key[0]} {key[1]}"] = {
            "bars": len(bars),
            "search_bars": split[key],
            "held_back_from": str(bars.index[split[key]]),
            "buy_and_hold_search_pct": hold_pct(bars, 0, split[key]),
            "buy_and_hold_held_back_pct": hold_pct(bars, split[key], len(bars)),
            "search_profitable_after_costs": sum(r["return_pct"] > 0 for r in net),
            "search_profitable_before_costs": sum(r["return_pct"] > 0 for r in gross),
            "search_median_return_pct": float(
                np.median([r["return_pct"] for r in net])
            ),
            "search_median_return_before_costs_pct": float(
                np.median([r["return_pct"] for r in gross])
            ),
            "settings_with_min_trades": len(ranked),
            "best_by_sharpe": ranked[0],
            "top5": ranked[:5],
            "held_back": {"best": held[0], "default_9_21_200": held[1]},
            "by_year": by_year,
        }
    summary["seconds"] = round(time.time() - t0, 1)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "traditional_slow.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"done in {summary['seconds']} s")


if __name__ == "__main__":
    main()
