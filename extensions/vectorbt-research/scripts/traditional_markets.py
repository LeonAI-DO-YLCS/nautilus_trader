"""
The 3-EMA crossover as traditional trading on EURUSD and gold (XAUUSD).

Same rules, grid and steps as scripts/traditional.py on Volatility 75: stop and reverse on the
1-minute close crossing all three EMAs, fill at the next minute's open, settings searched on the
first 70% of a span and the best tested once on the last 30%, then the best and 9/21/200 run on
every year of the Volatility 75 dataset's span (2016-08-19 to 2026-08-18), before and after
costs. Bars come from scripts/histdata_m1.py (bid prices, UTC). The files carry no spread, so
every order pays half of a fixed spread: 1.0 pip on EURUSD (a typical standard account) and
0.10 USD on gold (the median in our Pepperstone MT5 export). Swap is not included.

Example:
    python scripts/traditional_markets.py --data ~/data/histdata --start 2026-02-16 \
        --end 2026-08-17 --output /tmp/markets
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

SPREAD = {"EURUSD": 0.0001, "XAUUSD": 0.10}
SPAN = ("2016-08-19", "2026-08-18")
ALL: dict = {}


def load(data, pair):
    bars = pd.read_parquet(Path(data).expanduser() / f"{pair}_m1.parquet")
    bars = bars.set_index("time").loc[SPAN[0] : SPAN[1]]
    return pd.DataFrame(
        {"open": bars["open"], "close": bars["close"], "half_spread": SPREAD[pair] / 2},
        index=bars.index,
    )


def _year(args):
    year, settings = args
    bars = ALL["bars"].loc[str(year)]
    return year, t.backtest(bars, settings), t.backtest(bars, settings, fees=False)


def run(bars, start, end, workers):
    t0 = time.time()
    six = bars[
        (bars.index >= pd.Timestamp(start, tz="UTC"))
        & (bars.index < pd.Timestamp(end, tz="UTC"))
    ]
    t.BARS["six"] = six
    ALL["bars"] = bars
    split = int(len(six) * 0.7)
    settings = [s for s in itertools.product(*t.GRID.values()) if s[0] < s[1] < s[2]]
    chunks = [settings[i::workers] for i in range(workers)]
    with mp.get_context("fork").Pool(workers) as pool:
        search = [
            r
            for rows in pool.map(
                t._search_chunk, [(c, (0, split), True) for c in chunks]
            )
            for r in rows
        ]
        gross = [
            r
            for rows in pool.map(
                t._search_chunk, [(c, (0, split), False) for c in chunks]
            )
            for r in rows
        ]
        ranked = sorted(
            (r for r in search if r["trades"] >= 100),
            key=lambda r: r["sharpe"],
            reverse=True,
        )
        best = tuple(ranked[0]["setting"])
        held = t.backtest(six, [best, t.DEFAULT], (split, len(six)))
        years = sorted(set(bars.index.year))
        per_year = pool.map(_year, [(y, [best, t.DEFAULT]) for y in years])
    return {
        "span": [start, end],
        "bars": len(bars),
        "search_bars": split,
        "held_back_bars": len(six) - split,
        "settings": len(settings),
        "workers": workers,
        "seconds": round(time.time() - t0, 1),
        "search_profitable_after_costs": sum(r["return_pct"] > 0 for r in search),
        "search_profitable_before_costs": sum(r["return_pct"] > 0 for r in gross),
        "search_median_return_pct": float(np.median([r["return_pct"] for r in search])),
        "search_median_return_before_costs_pct": float(
            np.median([r["return_pct"] for r in gross])
        ),
        "best_by_sharpe": ranked[0],
        "top5": ranked[:5],
        "held_back": {"best": held[0], "default_9_21_200": held[1]},
        "by_year": {
            int(y): {"best": net[0], "default": net[1], "best_before_costs": raw[0]}
            for y, net, raw in per_year
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", nargs="+", default=list(SPREAD))
    args = parser.parse_args()
    workers = max(1, int(os.cpu_count() * 0.8))
    summary = {
        "costs": "half of a fixed spread per side; swap not included",
        "spread": SPREAD,
        "years_span": SPAN,
    }
    for pair in args.pairs:
        summary[pair] = run(load(args.data, pair), args.start, args.end, workers)
        print(pair, json.dumps(summary[pair], indent=2), flush=True)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "traditional_markets.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
